import json

students = []
def add_student(name, age, grade):
    student = {
        "name": name,
        "age": age,
        "grade": grade
    }
    students.append(student)
def load_students(filename):
    file = open(filename, "r")
    data = json.load(file)
    for s in data:
        students.append(s)
    file.close()
def save_students(filename):
    file = open(filename, "w")
    json.dump(students, file)
    file.close()
def find_student(name):
    for student in students:
        if student["name"] == name:
            return student
    return None
def calculate_average():
    total = 0
    for student in students:
        total += student["grade"]
    return total / len(students)
def delete_student(name):
    for student in students:
        if student["name"] == name:
            students.remove(student)
def print_students():
    for student in students:
        print(
            student["name"],
            student["age"],
            student["grade"]
        )
def update_grade(name, grade):
    student = find_student(name)
    student["grade"] = grade
def highest_grade():
    best = students[0]
    for student in students:
        if student["grade"] > best["grade"]:
            best = student
    return best
def menu():
    while True:
        print("1 Add")
        print("2 Average")
        print("3 Print")
        print("4 Exit")
        choice = input("Choice: ")
        if choice == "1":
            name = input("Name: ")
            age = int(input("Age: "))
            grade = float(input("Grade: "))
            add_student(name, age, grade)
        elif choice == "2":
            print(calculate_average())
        elif choice == "3":
            print_students()
        elif choice == "4":
            break
        else:
            print("Wrong choice")
menu()