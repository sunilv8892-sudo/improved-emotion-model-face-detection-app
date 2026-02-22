/// Student model representing a person enrolled in the system
class Student {
  final int? id;
  final String name;
  final String rollNumber;
  final String className;
  final String gender;
  final int age;
  final String phoneNumber;
  final DateTime enrollmentDate;

  Student({
    this.id,
    required this.name,
    required this.rollNumber,
    required this.className,
    required this.gender,
    required this.age,
    required this.phoneNumber,
    DateTime? enrollmentDate,
  }) : enrollmentDate = enrollmentDate ?? DateTime.now();

  /// Convert Student to JSON for database storage
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'roll_number': rollNumber,
      'class': className,
      'gender': gender,
      'age': age,
      'phone_number': phoneNumber,
      'enrollment_date': enrollmentDate.toIso8601String(),
    };
  }

  /// Create Student from database map
  factory Student.fromMap(Map<String, dynamic> map) {
    return Student(
      id: map['id'] as int?,
      name: map['name'] as String,
      rollNumber: map['roll_number'] as String,
      className: map['class'] as String,
      gender: map['gender'] as String? ?? '',
      age: map['age'] as int? ?? 0,
      phoneNumber: map['phone_number'] as String? ?? '',
      enrollmentDate: DateTime.parse(map['enrollment_date'] as String),
    );
  }

  @override
  String toString() =>
      'Student(id: $id, name: $name, rollNumber: $rollNumber, class: $className, gender: $gender, age: $age, phone: $phoneNumber)';
}
