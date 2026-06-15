from dataclasses import dataclass
import os
from pathlib import Path
from time import time
from typing import Union, Dict, Tuple, Optional

import cv2
import numpy as np

from ..pipeline_modules import *
from ..processing.russia_field_formats import (
    format_date_ddmmyyyy,
    format_department_code,
    format_series_number,
)


@dataclass(init=False)
class OCROptionsClass:
    """Class for storing OCR options for different document types.

    Holds common OCR options like fields needed, split preferences, etc.
    Sub-classes implement options specific to document types.
    """

    """list: Fields that need to be split for this doc type."""
    needed_split = []

    """list: English fields to recognize for this doc type."""
    en_fields = []

    """list: Russian fields to recognize for this doc type."""
    ru_fields = []

    """bool: Whether this doc type needs license number rotation."""
    needs_licence_rotation = False


    @classmethod
    def make_options(cls, country):
        """Factory method to make OCR options for a document type.

        Args:
            country (str): Country string (russia/belarus/usa)

        Returns:
            OCROptionsClass instance with options for the document type.
        """
        if country.lower() == 'belarus':
            return OCROptionsBelarus()
        elif country.lower() == 'russia':
            return OCROptionsEXTPassport()
        elif country.lower() == 'usa':
            return OCROptionsUSA()
        return OCROptionsClass()

class OCROptionsINTPassport(OCROptionsClass):
    """OCR options for internal Russian passports."""

    needed_split = ["Licence_number",
                    "Birth_place_ru", "Issue_organization_ru",
                    ]

    en_fields = ["Licence_number", "Issue_date", "Expiration_date", "Birth_date", "Issue_organisation_code", ]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru"]
    needs_licence_rotation = True

class OCROptionsEXTPassport(OCROptionsClass):
    """OCR options for external Russian passports."""

    # Длинные многословные поля — через WordsDetector, иначе линейный OCR склеивает слова без пробелов.
    needed_split = [
        "Licence_number",
        "Birth_place_ru",
        "Birth_place_en",
        "Issue_organization_ru",
        "Issue_organization_en",
        "Living_region_ru",
        "Living_region_en",
    ]

    # Серия и номер на развороте идут вертикально — перед OCR поворачиваем патч (как у внутреннего паспорта).
    needs_licence_rotation = True

    en_fields = ["Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                 "Expiration_date", "Birth_date", "Birth_place_en",
                 "Issue_organization_en", "Living_region_en", "Sex_en",
                 "Issue_organisation_code", "Middle_name_en"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru"]


class OCROptionsDL(OCROptionsClass):
    """OCR options for Russian driver's licenses."""

    needed_split = ["Licence_number", "Driver_class", "Birth_place_ru", "Birth_place_en",
                    "Living_region_ru", "Living_region_en", ]
    en_fields = ["Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                 "Expiration_date", "Driver_class", "Birth_date", "Birth_place_en",
                 "Issue_organization_en", "Living_region_en",  "Issue_organisation_code", "Middle_name_en"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", ]

class OCROptionsSNILS(OCROptionsClass):
    """OCR options for Russian SNILS documents."""

    needed_split = ["Last_name_ru", "First_name_ru", "Licence_number", "Issue_date",
                    "Birth_date", "Birth_place_ru", "Middle_name_ru", "Sex_ru", ]
    en_fields = ["Licence_number", "Issue_date", "Birth_date"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Middle_name_ru", "Sex_ru", ]

class OCROptionsBelarus(OCROptionsClass):
    """OCR options for Belarus passports.
    
    Все поля на английском языке, кроме photo, signature, signature2.
    Эти поля не распознаются (пропускаются в OCR).
    """
    
    needed_split = ["place_of_birth", "authority", "authority2", "nationality"]
    
    en_fields = ["passport_no", "surname", "names", "nationality", 
                 "date_of_birth", "identification_no", "sex", "place_of_birth",
                 "date_of_issue", "date_of_expiry", "authority", "authority2",
                 "type", "code_of_issuing"]
    
    ru_fields = []
    
    needs_licence_rotation = False


class OCROptionsUSA(OCROptionsClass):
    """OCR options for USA driver's licenses.

    Keys match the YOLO class order from docs_generator/usa/generator.py:
      IMAGE_FIELDS (photo, mini_photo, handwritten_signature) +
      TEXT_FIELDS (class..dob), without dob_short.
    UI labels: webapp/templates/index.html.
    photo/mini_photo/handwritten_signature в OCR не передаются.
    """
    
    needed_split = []
    
    en_fields = [
        "class",
        "end",
        "rest",
        "firstname",
        "lastname",
        "address",
        "sex",
        "hgt",
        "wgt",
        "eyes",
        "hair",
        "dd",
        "dln",
        "iss",
        "iss_duplicate",
        "exp",
        "dob",
    ]
    
    ru_fields = []
    
    needs_licence_rotation = False

class PipelineResults:
    """Stores results and metadata from a model pipeline.

    Attributes:
        meta_results (dict): Metadata from pipeline stages
        _timings (dict): Timing measurements for stages

    """
    def __init__(self):
        """Initializes empty result storage."""

        self.meta_results = dict(Quality={})
        self._timings = dict()

    @property
    def ocr(self) -> Union[Dict, None]:
        """Gets OCR extraction results dict, if available."""
        if self.meta_results.get('OCR'):
            return self.meta_results.get('OCR')
        else:
            return None

    @property
    def doctype(self) -> Union[str, None]:
        """Gets detected document type, if available."""
        doctype = self.meta_results.get('DocType')
        return doctype

    @property
    def quality(self) -> dict:
        """Gets image quality measurements."""
        return self.meta_results['Quality']

    @property
    def rotated_image(self) -> np.ndarray:
        """Gets image rotated by the Angle90 stage."""
        return self.meta_results['Angle90']['warped_img']

    @property
    def img_with_fixed_perspective(self) -> Union[list, None]:
        """Get result from doc detection net"""
        if self.meta_results.get('DocDetector'):
            return self.meta_results['DocDetector']['warped_img']
        else:
            return None

    @property
    def text_fields(self) -> Union[Tuple[list, list], None]:
        """Get text field patches with their meta"""
        for key in ('TextFieldsDetectorRussia', 'TextFieldsDetectorBelarus', 'TextFieldsDetectorUSA'):
            if self.meta_results.get(key):
                return self.meta_results[key]['bbox'], self.meta_results[key]['warped_img']
        return None

    @property
    def text_fields_meta(self) -> Union[Dict, None]:
        """Get text field meta"""
        for key in ('TextFieldsDetectorRussia', 'TextFieldsDetectorBelarus', 'TextFieldsDetectorUSA'):
            if self.meta_results.get(key):
                return self.meta_results[key]
        return None

    @property
    def words_patches(self) -> Union[Dict, None]:
        """Get split words patches"""
        if self.meta_results.get('WordsDetector'):
            return self.meta_results['WordsDetector']
        else:
            return None

    @property
    def passport_page_type(self) -> Optional[str]:
        """Gets passport page type result, if available."""
        ppt = self.meta_results.get('PassportPageType')
        if ppt and isinstance(ppt, dict):
            return ppt.get('page_type')
        return None

    @property
    def seal_info(self) -> Optional[Dict]:
        """Gets seal detection summary (without heavy image/mask data)."""
        raw = self.meta_results.get('PassportSealDetector')
        if not raw or not isinstance(raw, dict):
            return None
        return {
            'total_seals': raw.get('total_seals', 0),
            'has_seals': raw.get('has_seals', False),
            'last_seal': self._strip_seal(raw.get('last_seal')),
            'seals': [self._strip_seal(s) for s in raw.get('seals', [])],
        }

    @staticmethod
    def _strip_seal(seal: Optional[Dict]) -> Optional[Dict]:
        """Return seal dict without heavy numpy arrays (mask, images)."""
        if seal is None:
            return None
        return {k: v for k, v in seal.items()
                if k not in ('mask',)}

    @property
    def full_report(self) -> dict:
        """Returns full report in dict format"""
        summary_dict = {}
        summary_dict['DocType'] = self.doctype
        summary_dict['PassportPageType'] = self.passport_page_type
        summary_dict['SealInfo'] = self.seal_info
        summary_dict['OCR'] = self.ocr
        summary_dict['Quality'] = self.quality
        summary_dict['Timings'] = self.timings
        return summary_dict

    @property
    def timings(self) -> dict:
        """Gets per stage timings and total time."""
        total_time = 0
        timings = self._timings.copy()
        for value in timings.values():
            total_time += value
        timings['total'] = total_time
        return timings

    @timings.setter
    def timings(self, value):
        """Sets updated timings."""
        self._timings = self._timings | value




class Pipeline:
    """Pipeline for OCR processing of documents.

    Performs steps of pre-processing, text detection and OCR
    to extract text from documents.
    """

    def __init__(self, model_format='ONNX', device='cpu', verbose=False):
        """
        Initialize pipeline.

        Args:
            model_format (str): Format of models to use - ONNX, OpenVINO etc.
            device (str): Device for model inference - cpu, gpu etc.
            verbose (bool): Whether to print debug information.
        """
        # print(f'DEVICE: {device}')
        self.model_format = model_format
        self.device = device
        self.verbose = verbose
        # Базовые модели, которые нужны для любого документа
        self.angle90 = Angle90(model_format=model_format, device=device, verbose=verbose)
        self.doctype = DocType(model_format=model_format, device=device, verbose=verbose)
        self.doc_detector = DocDetector(model_format='PT', device=device, verbose=verbose)
        self.words_detector = WordsDetector(model_format=model_format, device=device, verbose=verbose)
        self.ocr_ru = OCRRus(
            model_format='ONNX' if model_format == 'OpenVINO' else model_format,
            device=device,
            verbose=verbose,
        )
        self.ocr_en = OCREngNums(
            model_format='ONNX' if model_format == 'OpenVINO' else model_format,
            device=device,
            verbose=verbose,
        )
        self.lcd_spoofing = LCDSpoofing(model_format=model_format, device=device, verbose=verbose)
        self.print_spoofing = PrintSpoofing(model_format=model_format, device=device, verbose=verbose)
        self.glare = Glare(model_format=model_format, device=device, verbose=verbose)
        self.blur = Blur(model_format=model_format, device=device, verbose=verbose)
        self.ocr_options = OCROptionsClass

        # Страно-специфичные модели подгружаем лениво после определения doc_type
        self.text_fields = None  # TextFieldsDetectorRussia / Belarus / USA
        self._text_fields_country = None  # 'russia' | 'belarus' | 'usa'
        self._usa_state = None
        self._passport_page_type_models = {}   # country -> PassportPageType
        self._seal_detector_models = {}        # country -> PassportSealDetector

    def reset_text_fields_detector(self) -> None:
        """Сброс кэша детектора полей (для batch-тестов — свой детектор на каждый кадр)."""
        self.text_fields = None
        self._text_fields_country = None
        self._usa_state = None

    def __call__(self, img_path: Union[Path, str, np.ndarray],
                 ocr=True,
                 get_doc_borders=True,
                 find_text_fields=True,
                 check_quality=True,
                 low_quality=True,
                 docconf=0.5,
                 img_size=1500,
                 reset_text_fields: bool = False,
                 ) -> PipelineResults:
        """
        Main pipeline processing method.

        Args:
            img_path: Path to input image.
            ocr: Whether to perform OCR.
            get_doc_borders: Whether to detect document borders.
            find_text_fields: Whether to detect text fields.
            check_quality: Whether to check image quality.
            low_quality: Whether to process low quality images.
            docconf: Минимальная уверенность DocType (0–1); ниже — остановка пайплайна при low_quality=False.
            img_size: Resize image to this size for processing.
            reset_text_fields: Пересоздать TextFieldsDetector на этом кадре (batch-тесты).

        Returns:
            PipelineResults with extracted information.
        """
        if reset_text_fields:
            self.reset_text_fields_detector()

        self.results = PipelineResults()

        img = self._prepare_image(img_path, img_size=img_size)

        self.time_measure = {}

        #getting angles
        self._model_call(self._angle, img)
        img = self.results.rotated_image

        #detecting doc (рамки)
        if get_doc_borders:
            self._model_call(self._doc_detector, img)
            img = self.results.img_with_fixed_perspective

        #getting doctype и conf (теперь после рамок)
        self._model_call(self._doctype, img)
        full_doc_type = self.results.doctype
        if full_doc_type == 'NONE':
            print("[!] The document on picture has unknown type")
            return self.results

        country, state = self._parse_doc_type(full_doc_type)
        self.ocr_options = self.ocr_options.make_options(country)

        self._init_text_fields(country, state)

        # PassportPageType: для russia и belarus, модели кэшируются по стране
        if country in ('russia', 'belarus'):
            if country not in self._passport_page_type_models:
                self._passport_page_type_models[country] = PassportPageType(
                    model_format=self.model_format,
                    device=self.device,
                    verbose=self.verbose,
                    country=country,
                )
            ppt_model = self._passport_page_type_models[country]
            ppt_result = ppt_model.predict(img)
            self.results.meta_results['PassportPageType'] = {'page_type': ppt_result}

        #getting quality
        if check_quality:
            self._model_call(self._glare, img)
            self._model_call(self._blur, img)
            self._model_call(self._print_spoofing, img)
            self._model_call(self._lcd_spoofing, img)

        # checking quality of doc
        if not low_quality:
            quality = self.results.quality
            # docconf — минимально допустимая уверенность классификатора типа документа (чем выше, тем увереннее).
            if (
                quality.get('Glare', False) == 'bad'
                or quality.get('Blur', False) == 'bad'
                or quality.get('DocConf', 0.0) < docconf
            ):
                return self.results

        page_type = self.results.meta_results.get('PassportPageType', {}).get('page_type')

        # Детектор печатей — для страниц с печатями (passport_pages) у russia и belarus
        if page_type == 'passport_pages' and country in ('russia', 'belarus'):
            if country not in self._seal_detector_models:
                self._seal_detector_models[country] = PassportSealDetector(
                    model_format='PT',
                    device=self.device,
                    verbose=self.verbose,
                    country=country,
                )
            self.seal_detector = self._seal_detector_models[country]
            self._model_call(self._passport_seal_detector, img)
        # OCR — для центральных разворотов паспорта (russia/belarus) и для usa
        elif (page_type == 'passport_centerfold'
              or country == 'usa'):
            if find_text_fields:
                rotate_licence = self.ocr_options.needs_licence_rotation
                self._model_call(self._fields_detector, img, rotate_licence=rotate_licence)
                text_fields = self.results.text_fields_meta
            else:
                return self.results

            if text_fields:
                self._model_call(self._split_words, text_fields.copy(), country)
                words_splitted = self.results.words_patches

                if ocr and words_splitted:
                    self._model_call(self._ocr, words_splitted, country)

        return self.results


    @staticmethod
    def _parse_doc_type(full_doc_type: str):
        """Parse full doc type string into (country, state/subtype).

        Examples:
            'usa_alabama' -> ('usa', 'alabama')
            'russia_passport' -> ('russia', None)
            'belarus_passport_1996' -> ('belarus', None)
        """
        ft = full_doc_type.lower()
        if ft.startswith('usa_'):
            return 'usa', ft[4:]
        elif ft.startswith('russia'):
            return 'russia', None
        elif ft.startswith('belarus'):
            return 'belarus', None
        return ft, None

    def _init_text_fields(self, country: str, state: str = None):
        """Lazily initialize the appropriate TextFieldsDetector.

        For USA, a new detector is created per state (since each state
        has its own model). For Russia/Belarus, the detector is reused.

        При смене страны детектор пересоздаётся — иначе после ошибочного
        DocType (usa_*) на seedream-снимке все следующие russia_passport
        в batch-прогоне шли бы через TextFieldsDetectorUSA.
        """
        if country == 'usa':
            if self._text_fields_country != 'usa' or self._usa_state != state:
                from ..pipeline_modules.textfields_detector import TextFieldsDetectorUSA
                self.text_fields = TextFieldsDetectorUSA(
                    state=state,
                    model_format='PT',
                    device=self.device,
                    verbose=self.verbose,
                )
                self._text_fields_country = 'usa'
                self._usa_state = state
        elif country == 'belarus':
            if self._text_fields_country != 'belarus':
                from ..pipeline_modules.textfields_detector import TextFieldsDetectorBelarus
                self.text_fields = TextFieldsDetectorBelarus(
                    model_format=self.model_format,
                    device=self.device,
                    verbose=self.verbose,
                )
                self._text_fields_country = 'belarus'
        elif country == 'russia':
            if self._text_fields_country != 'russia':
                from ..pipeline_modules.textfields_detector import TextFieldsDetector
                self.text_fields = TextFieldsDetector(
                    model_format=self.model_format,
                    device=self.device,
                    verbose=self.verbose,
                )
                self._text_fields_country = 'russia'

    def _angle(self, img):
        """
        Detect and fix angle of image.

        Args:
            img: Input image as numpy array.

        Returns:
            np.ndarray: Image with fixed angle.
        """
        result = self.angle90.predict_transform(img)
        self.results.meta_results = self.results.meta_results | result
        # return result[self.angle90.model_name]['warped_img']

    def _doctype(self, img):
        """
        Detect document type and confidence.

        Args:
            img: Input image

        Returns:
            str: Detected document type
            float: Confidence score
        """
        result = self.doctype.predict(img)
        doc_type = result[self.doctype.model_name]['doc_type']
        confidence = result[self.doctype.model_name]['confidence']

        # Если детектор границ документа не нашёл адекватную область (или нашёл слишком маленькую),
        # то DocType может быть "переуверенным" на нерелевантном изображении.
        # В этом случае принудительно считаем тип неизвестным.
        try:
            dd = self.results.meta_results.get("DocDetector") or {}
            bboxes = dd.get("bbox") or []
            # bbox в формате [x1,y1,x2,y2,...]; берём максимальную площадь
            if isinstance(bboxes, (list, tuple)) and len(bboxes) > 0 and hasattr(img, "shape"):
                h, w = img.shape[:2]
                img_area = float(max(1, h * w))
                areas = []
                for b in bboxes:
                    if b is None or len(b) < 4:
                        continue
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    areas.append(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
                max_frac = (max(areas) / img_area) if areas else 0.0
                min_frac = float(os.getenv("KYC_DOC_BBOX_MIN_FRAC", "0.12"))
                if max_frac < min_frac:
                    doc_type = "NONE"
                    confidence = 0.0
        except Exception:
            pass
        self.results.meta_results['DocType'] = doc_type
        self.results.meta_results['Quality']['DocConf'] = confidence
        return doc_type

    def _glare(self, img):
        """Check for glare quality"""
        qual, coef = self.glare.predict(img)[self.glare.model_name]
        self.results.meta_results['Quality']['Glare'] = qual
        return qual

    def _blur(self, img):
        """Check for blur quality."""
        qual, coef = self.blur.predict(img)[self.blur.model_name]
        self.results.meta_results['Quality']['Blur'] = qual
        return qual

    def _print_spoofing(self, img):
        """Check for print spoofing."""
        qual, coef = self.print_spoofing.predict(img)[self.print_spoofing.model_name]
        self.results.meta_results['Quality']['PrintSpoofing'] = qual
        return qual

    def _lcd_spoofing(self, img):
        """Check for LCD spoofing."""
        qual, coef = self.lcd_spoofing.predict(img)[self.lcd_spoofing.model_name]
        self.results.meta_results['Quality']['LCDSpoofing'] = qual
        return qual



    def _doc_detector(self, img):
        """
        Detect document borders and fix perspective.

        Args:
            img: Input image

        Returns:
            np.ndarray: Image with fixed perspective
        """
        result = self.doc_detector.predict_transform(img)
        self.results.meta_results = self.results.meta_results | result
        # img = result[self.doc_detector.model_name]['warped_img']
        # return img

    def _fields_detector(self, img, rotate_licence=False):
        """
        Detect text fields in document.

        Args:
            img: Input image
            rotate_licence: Whether to rotate license field

        Returns:
            dict: Detected text fields and patches
        """
        result = self.text_fields.predict_transform(img)
        text_fields = result[self.text_fields.model_name]


        if rotate_licence:
            for i, field in enumerate(text_fields['bbox']):
                if field[-1] == 'Licence_number':
                    text_fields['warped_img'][i] = cv2.rotate(text_fields['warped_img'][i],
                                                              cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.results.meta_results = self.results.meta_results | result

    @staticmethod
    def _pick_single_licence_number_bbox(bboxes, patches):
        """Оставляет один bbox класса Licence_number — с максимальной confidence (элемент с индексом 4)."""
        if not bboxes or len(bboxes) != len(patches):
            return bboxes, patches
        ln_idx = [i for i, b in enumerate(bboxes) if len(b) > 5 and b[-1] == 'Licence_number']
        if len(ln_idx) <= 1:
            return bboxes, patches

        def conf_at(i):
            try:
                return float(bboxes[i][4])
            except (TypeError, ValueError, IndexError):
                return 0.0

        best = max(ln_idx, key=conf_at)
        new_b, new_p = [], []
        for i, (b, p) in enumerate(zip(bboxes, patches)):
            if len(b) > 5 and b[-1] == 'Licence_number' and i != best:
                continue
            new_b.append(b)
            new_p.append(p)
        return new_b, new_p

    def _split_words(self, text_fields: dict, doc_type:str):
        """
        Split text fields into words.

        Args:
            text_fields: Detected text fields

        Returns:
            dict: Text fields splitted into words
        """

        bboxes, patches = text_fields.values()
        bboxes, patches = self._pick_single_licence_number_bbox(list(bboxes), list(patches))

        result = {}
        for i, bbox in enumerate(bboxes):

            if bbox[-1] not in self.ocr_options.en_fields and bbox[-1] not in self.ocr_options.ru_fields:
                continue

            patch = patches[i] if i < len(patches) else None
            if (
                patch is None
                or not isinstance(patch, np.ndarray)
                or patch.ndim < 2
                or patch.shape[0] < 1
                or patch.shape[1] < 1
            ):
                continue

            if bbox[-1] in self.ocr_options.needed_split:
                words = self.words_detector.predict_transform(patch)[self.words_detector.model_name]["warped_img"]
                # Fallback: if words detector failed, keep original patch
                if not words:
                    words = [patch]
            else:
                words = [patch]

            if result.get(bbox[-1]):
                result[bbox[-1]]['patches'].extend(words)
            else:
                result[bbox[-1]] = {'patches': words,
                                    'ocr': []}

        self.results.meta_results[self.words_detector.model_name] = result
        return result

    def _ocr(self, words_dict: dict, doc_type:str):
        """
        Perform OCR on splitted words.

        Args:
            words: Text fields splitted into words

        Returns:
            dict: OCR text for input words
        """
        ocr_dict = {}
        for field_name, words in words_dict.items():
            ocred_words = []
            for i, word in enumerate(words['patches']):
                if doc_type == 'SNILS' and 'date' in field_name.lower() and i % 2 == 1 or \
                        field_name in self.ocr_options.ru_fields:
                    result = self.ocr_ru.predict(word)[self.ocr_ru.model_name]['ocr_output']
                    result = self.ocr_ru.fix_errors(field_type=field_name, text=result)
                    words['ocr'].append(result)
                    ocred_words.append(result)
                elif field_name in self.ocr_options.en_fields:
                    result = self.ocr_en.predict(word)[self.ocr_en.model_name]['ocr_output']
                    result = self.ocr_en.fix_errors(field_type=field_name, text=result)
                    words['ocr'].append(result)
                    ocred_words.append(result)


            chunk = ' '.join(ocred_words).strip()
            is_russia = str(doc_type).lower() == 'russia'

            if field_name == 'Licence_number' and is_russia:
                formatted = format_series_number(chunk)
                ocr_dict[field_name] = formatted or ''
            elif field_name == 'Licence_number':
                # Загран / прочие: только цифры, шаблон «2 + 2 + 6» с пробелами; O часто путают с нулём.
                chunk = chunk.replace('O', '0').replace('o', '0')
                digits = ''.join(c for c in chunk if c.isdigit())
                if len(digits) >= 10:
                    d = digits[:10]
                    ocr_dict[field_name] = f'{d[:2]} {d[2:4]} {d[4:10]}'
                elif len(digits) >= 4:
                    ocr_dict[field_name] = f'{digits[:2]} {digits[2:4]} {digits[4:]}'.strip()
                else:
                    ocr_dict[field_name] = digits
            elif field_name == 'Issue_organisation_code' and is_russia:
                formatted = format_department_code(chunk)
                ocr_dict[field_name] = formatted or ''
            elif field_name == 'Issue_organisation_code':
                chunk = chunk.replace('O', '0').replace('o', '0')
                digits = ''.join(c for c in chunk if c.isdigit())
                if len(digits) >= 6:
                    d = digits[:6]
                    ocr_dict[field_name] = f'{d[:3]}-{d[3:6]}'
                else:
                    ocr_dict[field_name] = digits
            elif field_name in ('Birth_date', 'Issue_date') and is_russia:
                date_raw = '.'.join(ocred_words).strip() or chunk
                formatted = format_date_ddmmyyyy(date_raw)
                ocr_dict[field_name] = formatted or ''
            elif 'date' in field_name.lower() and doc_type == 'SNILS':
                ocr_dict[field_name] = ' '.join(ocred_words)
            elif 'date' in field_name.lower() and doc_type.lower() == 'belarus':
                # Для belarus даты в формате "28 08 1984" (с пробелами)
                # Если дата пришла как одно число (DDMMYYYY), разбиваем на DD MM YYYY
                date_text = ' '.join(ocred_words).strip()
                # Убираем все пробелы и проверяем, что это 8 цифр
                date_digits = ''.join(filter(str.isdigit, date_text))
                if len(date_digits) == 8:
                    # Формат DDMMYYYY -> DD MM YYYY
                    ocr_dict[field_name] = f'{date_digits[:2]} {date_digits[2:4]} {date_digits[4:]}'
                else:
                    # Если уже есть пробелы или другой формат, оставляем как есть
                    ocr_dict[field_name] = date_text
            elif 'date' in field_name.lower():
                ocr_dict[field_name] = '.'.join(ocred_words)
            else:
                # Join multiple lines conservatively: use space; if ALREADY contains content ending with comma, add space only
                joiner = ' '
                chunk = ' '.join(ocred_words)
                prev = ocr_dict.get(field_name, '')
                if prev:
                    ocr_dict[field_name] = (prev + joiner + chunk).strip()
                else:
                    ocr_dict[field_name] = chunk

            ocr_dict[field_name] = ocr_dict[field_name].replace('  ', ' ').strip()

        # Для belarus: объединяем authority и authority2 в одно поле authority
        if doc_type.lower() == 'belarus':
            if 'authority' in ocr_dict and 'authority2' in ocr_dict:
                authority_combined = f"{ocr_dict['authority']} {ocr_dict['authority2']}".strip()
                ocr_dict['authority'] = authority_combined
                # Удаляем authority2, так как оно объединено с authority
                del ocr_dict['authority2']

        # saving both OCR clear result and OCR of each patch
        self.results.meta_results['OCR'] = ocr_dict
        # self.results.meta_results[self.words_detector.model_name] = words_dict

    def _passport_seal_detector(self, img):
        """
        Detect passport seals on regular passport pages.

        Args:
            img: Input image

        Returns:
            dict: Detected seals information
        """
        result = self.seal_detector.predict_transform(img)
        self.results.meta_results = self.results.meta_results | result

    def _model_call(self, func, *args, **kwargs):
        """ Wrapper for making timing calculations."""
        time_start = time()
        result = func(*args, **kwargs)
        self.results.timings = {func.__name__: round(time() - time_start, 4)}
        return result

    def _prepare_image(self, img_path: Union[Path, str, np.ndarray], img_size: int = 1500):
        """
        Load image from path, validate it and resize.

        Args:
            img_path: Path to input image.
            img_size: Resize image to this size.

        Returns:
            np.ndarray: Loaded and resized image.
        """

        if isinstance(img_path, Path):
            img = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results.meta_results['image_path'] = img_path.as_posix()
        elif isinstance(img_path, str):
            img_path = Path(img_path)
            img = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results.meta_results['image_path'] = img_path
        elif isinstance(img_path, np.ndarray):
            img = img_path
        else:
            raise Exception("Unsupported image type")

        # check size of image, and resize if above 1500
        h, w = img.shape[:2]
        ratio = max(max(h, w) / img_size, 1)
        new_h, new_w = int(h // ratio), int(w // ratio)
        img = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.results.meta_results['original_img'] = img

        return img










