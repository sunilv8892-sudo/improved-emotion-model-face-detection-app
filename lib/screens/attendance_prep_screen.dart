import 'package:flutter/material.dart';
import '../database/database_manager.dart';
import '../models/subject_model.dart';
import '../utils/constants.dart';
import '../widgets/animated_background.dart';
import 'attendance_screen.dart';

class AttendancePrepScreen extends StatefulWidget {
  const AttendancePrepScreen({super.key});

  @override
  State<AttendancePrepScreen> createState() => _AttendancePrepScreenState();
}

class _AttendancePrepScreenState extends State<AttendancePrepScreen> {
  late DatabaseManager _dbManager;
  final _teacherNameController = TextEditingController();
  final _newSubjectController = TextEditingController();
  
  List<Subject> _subjects = [];
  Subject? _selectedSubject;
  bool _isCreatingNewSubject = false;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      _dbManager = DatabaseManager();
      await _dbManager.database;
      await _loadSubjects();
      if (mounted) setState(() => _isLoading = false);
    } catch (e) {
      debugPrint('Init error: $e');
      if (mounted) {
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  Future<void> _loadSubjects() async {
    try {
      final subjects = await _dbManager.getAllSubjects();
      if (mounted) {
        setState(() {
          _subjects = subjects;
          if (_subjects.isNotEmpty) {
            _selectedSubject = _subjects.first;
          }
        });
      }
    } catch (e) {
      debugPrint('Error loading subjects: $e');
    }
  }

  Future<void> _createNewSubject(String name) async {
    if (name.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a subject name')),
      );
      return;
    }

    try {
      final newSubject = await _dbManager.getOrCreateSubject(name.trim());
      
      if (mounted) {
        setState(() {
          if (!_subjects.any((s) => s.id == newSubject.id)) {
            _subjects.add(newSubject);
          }
          _selectedSubject = newSubject;
          _isCreatingNewSubject = false;
          _newSubjectController.clear();
        });
        
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('✓ Subject created'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      debugPrint('Error creating subject: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  void _proceed() {
    final teacherName = _teacherNameController.text.trim();
    
    if (teacherName.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter teacher name')),
      );
      return;
    }

    if (_selectedSubject == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select or create a subject')),
      );
      return;
    }

    // Navigate to AttendanceScreen
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (context) => AttendanceScreen(
          teacherName: teacherName,
          subject: _selectedSubject!,
        ),
      ),
    );
  }

  @override
  void dispose() {
    _teacherNameController.dispose();
    _newSubjectController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(title: const Text('Setup')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Attendance Setup'),
        elevation: 0,
        centerTitle: true,
        flexibleSpace: Container(
          decoration: BoxDecoration(gradient: AppConstants.blueGradient),
        ),
      ),
      body: AnimatedBackground(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            // Header
            Text(
              'Teacher Information',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 24),

            // Teacher Name Field
            const Text(
              'Teacher Name',
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _teacherNameController,
              decoration: InputDecoration(
                hintText: 'Enter your name',
                filled: true,
                fillColor: AppConstants.inputFill,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: AppConstants.cardBorder),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: AppConstants.cardBorder),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(
                    color: AppConstants.primaryColor,
                    width: 2,
                  ),
                ),
                prefixIcon: const Icon(Icons.person),
                hintStyle: const TextStyle(color: AppConstants.textSecondary),
              ),
              style: const TextStyle(color: AppConstants.textPrimary),
            ),
            const SizedBox(height: 24),

            // Subject Header
            const Text(
              'Subject',
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),

            // Toggle between existing and new
            if (!_isCreatingNewSubject)
              Column(
                children: [
                  // Dropdown for existing subjects
                  if (_subjects.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      decoration: BoxDecoration(
                        color: AppConstants.inputFill,
                        border: Border.all(color: AppConstants.cardBorder),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: DropdownButton<Subject>(
                        value: _selectedSubject,
                        isExpanded: true,
                        underline: const SizedBox(),
                        items: _subjects
                            .map((subject) => DropdownMenuItem(
                              value: subject,
                              child: Text(
                                subject.name,
                                style: const TextStyle(
                                  color: AppConstants.textPrimary,
                                ),
                              ),
                            ))
                            .toList(),
                        onChanged: (Subject? newValue) {
                          setState(() {
                            _selectedSubject = newValue;
                          });
                        },
                      ),
                    ),
                  const SizedBox(height: 12),
                  // Create new button
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: () {
                        setState(() {
                          _isCreatingNewSubject = true;
                        });
                      },
                      icon: const Icon(Icons.add),
                      label: const Text('Create New Subject'),
                    ),
                  ),
                ],
              )
            else
              // Create new subject input
              Column(
                children: [
                  TextField(
                    controller: _newSubjectController,
                    decoration: InputDecoration(
                      hintText: 'Enter subject name',
                      filled: true,
                      fillColor: AppConstants.inputFill,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: const BorderSide(color: AppConstants.cardBorder),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: const BorderSide(color: AppConstants.cardBorder),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: const BorderSide(
                          color: AppConstants.primaryColor,
                          width: 2,
                        ),
                      ),
                      prefixIcon: const Icon(Icons.subject),
                      hintStyle: const TextStyle(color: AppConstants.textSecondary),
                    ),
                    style: const TextStyle(color: AppConstants.textPrimary),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: () {
                            _createNewSubject(_newSubjectController.text);
                          },
                          icon: const Icon(Icons.check),
                          label: const Text('Create'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () {
                            setState(() {
                              _isCreatingNewSubject = false;
                              _newSubjectController.clear();
                            });
                          },
                          icon: const Icon(Icons.close),
                          label: const Text('Cancel'),
                        ),
                      ),
                    ],
                  ),
                ],
              ),

            const SizedBox(height: 32),

            // Proceed Button
            ElevatedButton.icon(
              onPressed: _proceed,
              icon: const Icon(Icons.arrow_forward),
              label: const Text(
                'Start Attendance',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppConstants.primaryColor,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
