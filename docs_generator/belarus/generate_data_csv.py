import random
import csv
from datetime import datetime, timedelta

# Списки для генерации данных
surnames = [
    "POPOV", "SOKOLOV", "MOROZOV", "KARPENKO", "FADEEV", "GOLOVIN", "VASILIEV",
    "ANTONOV", "FEDOROV", "POPOVA", "BELIAEV", "NESTEROV", "VINOGRADOV", "SOKOLOVA",
    "EGOROV", "ORLOV", "KUZNETSOV", "DMITRIEV", "FROLOV", "BOGDANOV", "MIRONOV",
    "KOZLOV", "YAKOVLEV", "SIDOROV", "PAVLOV", "KIRILLOV", "SMIRNOVA", "KOVALENKO",
    "FILIPPOV", "VOLKOV", "SOLOVYOV", "NIKOLAEV", "PETROV", "IVANOV", "SMIRNOV"
]

names_male = [
    "IVAN", "PETR", "DMITRY", "ARTEM", "EGOR", "SERGEY", "ALEXEY", "NIKOLAI",
    "ANDREY", "MAXIM", "VLADIMIR", "ALEXANDER", "ROMAN", "MIKHAIL", "PAVEL"
]

names_female = [
    "NATALIA", "SVETLANA", "OLGA", "TATIANA", "ALINA", "MARIA", "ELENA", "ANNA",
    "VERA", "IRINA", "EKATERINA", "YULIA", "ANASTASIA", "DARYA", "SOFIA"
]

def generate_passport_no():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"{random.choice(letters)}{random.choice(letters)}{random.randint(1000000, 9999999)}"

def generate_identification_no():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits1 = "".join([str(random.randint(0, 9)) for _ in range(7)])
    letter1 = random.choice(letters)
    digits2 = "".join([str(random.randint(0, 9)) for _ in range(3)])
    letters2 = "".join([random.choice(letters) for _ in range(2)])
    return f"{digits1}{letter1}{digits2}{letters2}"

def generate_date(start_year=1970, end_year=2005):
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{day:02d} {month:02d} {year}"

def generate_issue_date(birth_date_str):
    # Парсим дату рождения
    parts = birth_date_str.split()
    birth_year = int(parts[2])
    # Дата выдачи: от 18 лет после рождения до текущего года
    min_issue_year = birth_year + 18
    max_issue_year = min(2024, birth_year + 50)
    if min_issue_year > max_issue_year:
        min_issue_year = max_issue_year - 10
    
    issue_year = random.randint(min_issue_year, max_issue_year)
    issue_month = random.randint(1, 12)
    issue_day = random.randint(1, 28)
    return f"{issue_day:02d} {issue_month:02d} {issue_year}"

def generate_expiry_date(issue_date_str):
    # Парсим дату выдачи и добавляем 10 лет
    parts = issue_date_str.split()
    issue_year = int(parts[2])
    expiry_year = issue_year + 10
    expiry_month = int(parts[1])
    expiry_day = int(parts[0])
    return f"{expiry_day:02d} {expiry_month:02d} {expiry_year}"

def generate_row():
    sex = random.choice(["M", "F"])
    surname = random.choice(surnames)
    if sex == "M":
        name = random.choice(names_male)
    else:
        name = random.choice(names_female)
    
    passport_no = generate_passport_no()
    date_of_birth = generate_date(1970, 2005)
    date_of_issue = generate_issue_date(date_of_birth)
    date_of_expiry = generate_expiry_date(date_of_issue)
    identification_no = generate_identification_no()
    
    return {
        "passport_no": passport_no,
        "surname": surname,
        "names": name,
        "nationality": "REPUBLIC OF BELARUS",
        "date_of_birth": date_of_birth,
        "identification_no": identification_no,
        "sex": sex,
        "place_of_birth": "BLR",
        "date_of_issue": date_of_issue,
        "date_of_expiry": date_of_expiry,
        "authority": "MINISTRY OF",
        "authority2": "INTERNAL AFFAIRS",
        "authority_code": str(random.randint(100, 999)),
        "type": "P",
        "code_of_issuing": "BLR"
    }

def main():
    output_file = "data/data.csv"
    num_rows = 550
    
    fieldnames = [
        "passport_no", "surname", "names", "nationality", "date_of_birth",
        "identification_no", "sex", "place_of_birth", "date_of_issue",
        "date_of_expiry", "authority", "authority2", "authority_code",
        "type", "code_of_issuing"
    ]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(num_rows):
            row = generate_row()
            writer.writerow(row)
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{num_rows} rows...")
    
    print(f"Generated {num_rows} rows in {output_file}")

if __name__ == "__main__":
    main()





