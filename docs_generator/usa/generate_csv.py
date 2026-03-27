import argparse
import csv
import random
import string
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


# Наборы значений из задания
CLASSES = ["D", "A", "B", "C", "M"]
CLASS_WEIGHTS = [0.8, 0.05, 0.05, 0.05, 0.05]  # D доминирует
SEX = ["M", "F", "X"]
SEX_WEIGHTS = [0.5, 0.495, 0.005]  # X крайне редко
EYES = ["BRO", "BLU", "GRN", "HAZ", "GRY"]
HAIR = ["BRO", "BLK", "BLN", "RED", "WHI", "BAL"]
END = ["NONE", "H", "N", "P"]
END_WEIGHTS = [0.9, 0.03, 0.04, 0.03]  # допуски редки
REST = ["NONE", "B", "G"]
REST_WEIGHTS = [0.9, 0.07, 0.03]  # ограничения редки

# Минимальный возраст для выдачи прав
MIN_LICENSE_AGE = 16
# Диапазон лет действия прав
EXP_YEARS_RANGE = (4, 8)
# Возрастные границы для генерации даты рождения
DOB_YEAR_RANGE = (1940, 2005)

# Компоненты адреса
DIRECTIONS = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
STREET_NAMES = [
    "MAIN", "WASHINGTON", "OAK", "5TH", "7TH", "10TH", "PARK", "ELM", "PINE",
    "MAPLE", "CENTER", "CHURCH", "FIRST", "SECOND", "THIRD", "FOURTH", "SIXTH",
    "EIGHTH", "NINTH", "BROADWAY", "LINCOLN", "JEFFERSON", "MADISON", "JACKSON",
    "ROOSEVELT", "KENNEDY", "MLK JR", "KING", "QUEEN", "STATE", "COUNTY",
    "KRESKY", "GRAND", "UNION", "SPRING", "SUMMER", "WINTER", "RIVER", "LAKE"
]
STREET_SUFFIXES = ["ST", "AVE", "RD", "BLVD", "LN", "DR", "CT", "PL", "WAY"]
UNIT_TYPES = ["APT", "STE", "UNIT"]

# Штаты с городами и первой цифрой ZIP кода
STATES_DATA = {
    "WA": {"zip_start": "9", "cities": ["SEATTLE", "SPOKANE", "TACOMA", "VANCOUVER", "BELLEVUE", "EVERETT", "KENT", "RENTON"]},
    "ID": {"zip_start": "8", "cities": ["BOISE", "NAMPA", "MERIDIAN", "IDAHO FALLS", "POCATELLO", "COEUR D ALENE", "TWIN FALLS", "CALDWELL"]},
    "CA": {"zip_start": "9", "cities": ["LOS ANGELES", "SAN DIEGO", "SAN JOSE", "SAN FRANCISCO", "FRESNO", "SACRAMENTO", "LONG BEACH", "OAKLAND"]},
    "NY": {"zip_start": "1", "cities": ["NEW YORK", "BUFFALO", "ROCHESTER", "YONKERS", "SYRACUSE", "ALBANY", "NEW ROCHELLE", "MOUNT VERNON"]},
    "TX": {"zip_start": "7", "cities": ["HOUSTON", "SAN ANTONIO", "DALLAS", "AUSTIN", "FORT WORTH", "EL PASO", "ARLINGTON", "CORPUS CHRISTI"]},
    "FL": {"zip_start": "3", "cities": ["JACKSONVILLE", "MIAMI", "TAMPA", "ORLANDO", "ST PETERSBURG", "HIALEAH", "TALLAHASSEE", "FORT LAUDERDALE"]},
    "IL": {"zip_start": "6", "cities": ["CHICAGO", "AURORA", "NAPERVILLE", "JOLIET", "ROCKFORD", "ELGIN", "PEORIA", "CHAMPAGN"]},
    "PA": {"zip_start": "1", "cities": ["PHILADELPHIA", "PITTSBURGH", "ALLENTOWN", "ERIE", "READING", "SCRANTON", "BETHLEHEM", "LANCASTER"]},
    "OH": {"zip_start": "4", "cities": ["COLUMBUS", "CLEVELAND", "CINCINNATI", "TOLEDO", "AKRON", "DAYTON", "PARMA", "CANTON"]},
    "GA": {"zip_start": "3", "cities": ["ATLANTA", "AUGUSTA", "COLUMBUS", "SAVANNAH", "ATHENS", "SANDY SPRINGS", "ROSWELL", "MACON"]},
}


@dataclass
class PersonRecord:
    class_code: str
    sex: str
    eyes: str
    hair: str
    end: str
    rest: str
    dln: str
    firstname: str
    lastname: str
    address: str
    dd: str
    dob: str
    iss: str
    exp: str
    hgt: str
    wgt: str

    def to_row(self):
        return [
            self.class_code,
            self.sex,
            self.eyes,
            self.hair,
            self.end,
            self.rest,
            self.dln,
            self.firstname,
            self.lastname,
            self.address,
            self.dd,
            self.dob,
            self.iss,
            self.exp,
            self.hgt,
            self.wgt,
        ]


def random_date(start: date, end: date) -> date:
    """Случайная дата между start и end включительно."""
    if start > end:
        start, end = end, start
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def generate_dln() -> str:
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = "".join(random.choices(string.digits, k=6))
    suffix = random.choice(string.ascii_uppercase)
    return f"{letters}{digits}{suffix}"


def generate_dd() -> str:
    length = random.randint(10, 15)
    return "".join(random.choices(string.digits, k=length))


def generate_height() -> str:
    feet = random.randint(4, 6)
    inches = random.randint(0, 11)
    return f"{feet}'-{inches:02d}\""


def generate_weight() -> str:
    return f"{random.randint(100, 260)} lb"


def generate_name() -> tuple[str, str]:
    # Небольшие наборы имён/фамилий для простоты
    firstnames = [
        "JAMES",
        "ROBERT",
        "JOHN",
        "MICHAEL",
        "WILLIAM",
        "DAVID",
        "RICHARD",
        "JOSEPH",
        "THOMAS",
        "CHARLES",
        "MARY",
        "PATRICIA",
        "LINDA",
        "BARBARA",
        "ELIZABETH",
        "JENNIFER",
        "MARIA",
        "SUSAN",
        "MARGARET",
        "DOROTHY",
    ]
    lastnames = [
        "SMITH",
        "JOHNSON",
        "WILLIAMS",
        "BROWN",
        "JONES",
        "GARCIA",
        "MILLER",
        "DAVIS",
        "RODRIGUEZ",
        "MARTINEZ",
        "HERNANDEZ",
        "LOPEZ",
        "GONZALEZ",
        "WILSON",
        "ANDERSON",
        "THOMAS",
        "TAYLOR",
        "MOORE",
        "JACKSON",
        "MARTIN",
    ]
    return random.choice(firstnames), random.choice(lastnames)


def generate_dates() -> tuple[str, str, str]:
    """Генерирует DOB, ISS, EXP с сохранением логики последовательности."""
    dob_year = random.randint(*DOB_YEAR_RANGE)
    dob = date(dob_year, random.randint(1, 12), random.randint(1, 28))

    # Минимально 16 лет к дате выдачи
    earliest_issue = dob + timedelta(days=MIN_LICENSE_AGE * 365)
    today = date.today()
    # Выбираем дату выдачи не позже сегодняшней и не раньше earliest_issue
    iss = random_date(earliest_issue, today)

    # EXP позже ISS на 4-8 лет
    exp_years = random.randint(*EXP_YEARS_RANGE)
    exp = iss + timedelta(days=exp_years * 365)

    def fmt(d: date) -> str:
        return d.strftime("%m/%d/%Y")

    return fmt(dob), fmt(iss), fmt(exp)


def weighted_choice(items: list[str], weights: list[float]) -> str:
    """Выбор элемента с указанными весами."""
    return random.choices(items, weights=weights, k=1)[0]


def generate_address() -> str:
    """Генерирует адрес по шаблону: {NUMBER} {DIRECTION} {STREET} {SUFFIX} {UNIT} {CITY} {STATE} {ZIP5}-{ZIP4}"""
    # NUMBER: 1-5 цифр, редко буквы (5% случаев)
    if random.random() < 0.05:
        number = f"{random.randint(1, 9999)}-{random.choice(string.ascii_uppercase)}"
    else:
        number = str(random.randint(1, 99999))
    
    # DIRECTION: опционально (25% случаев)
    direction = ""
    if random.random() < 0.25:
        direction = random.choice(DIRECTIONS) + " "
    
    # STREET: название улицы
    street = random.choice(STREET_NAMES)
    
    # SUFFIX: стандартное сокращение
    suffix = random.choice(STREET_SUFFIXES)
    
    # UNIT: опционально (15% случаев)
    unit = ""
    if random.random() < 0.15:
        unit_type = random.choice(UNIT_TYPES)
        # Может быть номер или буква
        if random.random() < 0.7:
            unit_num = str(random.randint(1, 999))
        else:
            unit_num = f"{random.randint(1, 99)}{random.choice(string.ascii_uppercase)}"
        unit = f" {unit_type} {unit_num}"
    
    # STATE и CITY: выбираем случайный штат и город из него
    state = random.choice(list(STATES_DATA.keys()))
    city = random.choice(STATES_DATA[state]["cities"])
    zip_start = STATES_DATA[state]["zip_start"]
    
    # ZIP5: 5 цифр, первая привязана к штату
    zip5 = zip_start + "".join(random.choices(string.digits, k=4))
    
    # ZIP4: 4 цифры после дефиса
    zip4 = "".join(random.choices(string.digits, k=4))
    
    # Собираем адрес
    address = f"{number} {direction}{street} {suffix}{unit} {city} {state} {zip5}-{zip4}"
    return address


def generate_class() -> str:
    """Генерирует класс(ы) прав. Может быть несколько классов через пробел."""
    rand = random.random()
    
    # 75% - только D (самый частый)
    if rand < 0.75:
        return "D"
    
    # 15% - D + M (легковой + мотоцикл, частая комбинация)
    elif rand < 0.90:
        return "D M"
    
    # 5% - только коммерческие классы (A, B, C)
    elif rand < 0.95:
        return random.choice(["A", "B", "C"])
    
    # 3% - коммерческий + D
    elif rand < 0.98:
        commercial = random.choice(["A", "B", "C"])
        return f"{commercial} D"
    
    # 2% - другие комбинации (D + коммерческий, коммерческий + M, и т.д.)
    else:
        options = [
            "D A",
            "D B", 
            "D C",
            "A M",
            "B M",
            "C M",
            "D A M",
        ]
        return random.choice(options)


def generate_record() -> PersonRecord:
    firstname, lastname = generate_name()
    dob, iss, exp = generate_dates()
    
    return PersonRecord(
        class_code=generate_class(),
        sex=weighted_choice(SEX, SEX_WEIGHTS),
        eyes=random.choice(EYES),
        hair=random.choice(HAIR),
        end=weighted_choice(END, END_WEIGHTS),
        rest=weighted_choice(REST, REST_WEIGHTS),
        dln=generate_dln(),
        firstname=firstname,
        lastname=lastname,
        address=generate_address(),
        dd=generate_dd(),
        dob=dob,
        iss=iss,
        exp=exp,
        hgt=generate_height(),
        wgt=generate_weight(),
    )


def write_csv(path: Path, rows: list[PersonRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Class",
        "Sex",
        "Eyes",
        "Hair",
        "End",
        "Rest",
        "DLN",
        "Firstname",
        "Lastname",
        "Address",
        "DD",
        "DOB",
        "ISS",
        "EXP",
        "HGT",
        "WGT",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row.to_row())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генератор CSV с персональными данными для водительских прав (USA)."
    )
    parser.add_argument(
        "-n",
        "--rows",
        type=int,
        default=10,
        help="Сколько строк сгенерировать (по умолчанию 10).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs_generator/usa/out/personal_data.csv"),
        help="Путь к результирующему CSV (по умолчанию docs_generator/usa/out/personal_data.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [generate_record() for _ in range(max(0, args.rows))]
    write_csv(args.output, rows)
    print(f"Создан файл: {args.output} (строк данных: {len(rows)})")


if __name__ == "__main__":
    main()
