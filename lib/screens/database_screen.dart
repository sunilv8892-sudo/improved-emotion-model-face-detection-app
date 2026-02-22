import 'package:flutter/material.dart';
import '../database/database_manager.dart';
import '../modules/m4_attendance_management.dart';
import '../models/attendance_model.dart';
import '../utils/constants.dart';
import '../widgets/animated_background.dart';

class DatabaseScreen extends StatefulWidget {
  const DatabaseScreen({super.key});

  @override
  State<DatabaseScreen> createState() => _DatabaseScreenState();
}

class _DatabaseScreenState extends State<DatabaseScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;
  late final DatabaseManager _dbManager;
  late final AttendanceManagementModule _attendanceModule;

  late Future<SystemStatistics> _systemStatsFuture;
  late Future<List<AttendanceDetails>> _studentDetailsFuture;
  late Future<List<AttendanceDetails>> _enrolledStudentsFuture;
  late Future<List<_DateAttendanceSummary>> _attendanceHistoryFuture;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _dbManager = DatabaseManager();
    _attendanceModule = AttendanceManagementModule(_dbManager);
    _reloadData();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _reloadData();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Attendance Dashboard'),
        flexibleSpace: Container(
          decoration: BoxDecoration(gradient: AppConstants.blueGradient),
        ),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(icon: Icon(Icons.bar_chart), text: 'Overview'),
            Tab(icon: Icon(Icons.people), text: 'Enrolled'),
          ],
        ),
      ),
      body: AnimatedBackground(
        child: TabBarView(
          controller: _tabController,
          children: [_buildOverviewTab(), _buildEnrolledStudentsTab()],
        ),
      ),
    );
  }

  Widget _buildOverviewTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(AppConstants.paddingMedium),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Statistics Header
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: AppConstants.primaryColor.withAlpha(26),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.bar_chart,
                  color: AppConstants.primaryColor,
                  size: 24,
                ),
              ),
              const SizedBox(width: AppConstants.paddingMedium),
              const Text(
                'System Statistics',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: AppConstants.paddingMedium),
          FutureBuilder<SystemStatistics>(
            future: _systemStatsFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState != ConnectionState.done) {
                return const Center(child: CircularProgressIndicator());
              }
              final stats = snapshot.data;
              if (stats == null) {
                return Container(
                  padding: const EdgeInsets.all(AppConstants.paddingLarge),
                  child: const Text('Statistics are not available yet.'),
                );
              }
              return _buildStatsCards(stats);
            },
          ),
          const SizedBox(height: AppConstants.paddingLarge),

          // Attendance History Header
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: AppConstants.successColor.withAlpha(26),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.history,
                  color: AppConstants.successColor,
                  size: 24,
                ),
              ),
              const SizedBox(width: AppConstants.paddingMedium),
              const Expanded(
                child: Text(
                  'Attendance History',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ),
              IconButton(
                onPressed: _reloadData,
                icon: const Icon(Icons.refresh, size: 20),
                color: AppConstants.primaryColor,
                tooltip: 'Refresh',
              ),
            ],
          ),
          const SizedBox(height: AppConstants.paddingSmall),
          FutureBuilder<List<_DateAttendanceSummary>>(
            future: _attendanceHistoryFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState != ConnectionState.done) {
                return const Center(child: CircularProgressIndicator());
              }
              final history = snapshot.data ?? [];
              if (history.isEmpty) {
                return Container(
                  padding: const EdgeInsets.all(AppConstants.paddingLarge),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(
                      AppConstants.borderRadius,
                    ),
                    border: Border.all(color: AppConstants.cardBorder),
                  ),
                  child: const Center(
                    child: Column(
                      children: [
                        Icon(
                          Icons.event_busy,
                          size: 40,
                          color: AppConstants.textTertiary,
                        ),
                        SizedBox(height: 8),
                        Text('No attendance records yet.'),
                        SizedBox(height: 4),
                        Text(
                          'Take attendance to see history here',
                          style: TextStyle(
                            color: AppConstants.textTertiary,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              }
              return ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: history.length,
                separatorBuilder: (context, index) =>
                    const SizedBox(height: AppConstants.paddingSmall),
                itemBuilder: (context, index) =>
                    _attendanceHistoryTile(history[index]),
              );
            },
          ),

          const SizedBox(height: AppConstants.paddingLarge),

          // Per-Student Attendance Summary
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: AppConstants.accentColor.withAlpha(26),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.person_search,
                  color: AppConstants.accentColor,
                  size: 24,
                ),
              ),
              const SizedBox(width: AppConstants.paddingMedium),
              const Text(
                'Student Attendance',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: AppConstants.paddingSmall),
          FutureBuilder<List<AttendanceDetails>>(
            future: _studentDetailsFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState != ConnectionState.done) {
                return const Center(child: CircularProgressIndicator());
              }
              final students = snapshot.data ?? [];
              if (students.isEmpty) {
                return Container(
                  padding: const EdgeInsets.all(AppConstants.paddingLarge),
                  child: const Text('No students enrolled yet.'),
                );
              }
              return ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: students.length,
                separatorBuilder: (context, index) =>
                    const SizedBox(height: AppConstants.paddingSmall),
                itemBuilder: (context, index) => _studentTile(students[index]),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildEnrolledStudentsTab() {
    return FutureBuilder<List<AttendanceDetails>>(
      future: _enrolledStudentsFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        final enrolledStudents = snapshot.data ?? [];
        if (enrolledStudents.isEmpty) {
          return Container(
            padding: const EdgeInsets.all(AppConstants.paddingLarge),
            child: const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.person_add,
                    size: 48,
                    color: AppConstants.textTertiary,
                  ),
                  SizedBox(height: AppConstants.paddingMedium),
                  Text('No enrolled students yet.'),
                  SizedBox(height: AppConstants.paddingSmall),
                  Text(
                    'Students need to complete face enrollment first',
                    style: TextStyle(
                      color: AppConstants.textTertiary,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
          );
        }
        return ListView(
          padding: const EdgeInsets.all(AppConstants.paddingMedium),
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AppConstants.primaryColor.withAlpha(26),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(
                    Icons.people,
                    color: AppConstants.primaryColor,
                    size: 24,
                  ),
                ),
                const SizedBox(width: AppConstants.paddingMedium),
                Text(
                  'Enrolled Students (${enrolledStudents.length})',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: AppConstants.paddingMedium),
            ...enrolledStudents.map((detail) => _enrolledStudentCard(detail)),
          ],
        );
      },
    );
  }

  Widget _enrolledStudentCard(AttendanceDetails detail) {
    final student = detail.student;
    final ratio = detail.totalClasses > 0
        ? '${detail.presentCount}/${detail.totalClasses}'
        : '0/0';
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(AppConstants.paddingMedium),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    student.name,
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: AppConstants.paddingSmall / 2),
                  Text(
                    'Roll: ${student.rollNumber} · Class: ${student.className}',
                    style: const TextStyle(fontSize: 12),
                  ),
                  const SizedBox(height: AppConstants.paddingSmall / 2),
                  Text(
                    'Enrolled: ${student.enrollmentDate.toLocal().toString().split(' ')[0]}',
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppConstants.textTertiary,
                    ),
                  ),
                  const SizedBox(height: AppConstants.paddingSmall / 2),
                  Text(
                    'Attendance: $ratio (${detail.attendancePercentage.toStringAsFixed(1)}%)',
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
            IconButton(
              icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
              tooltip: 'Delete student',
              onPressed: () => _handleDeleteStudent(detail),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatsCards(SystemStatistics stats) {
    final statItems = [
      _StatData(
        label: 'Students',
        value: stats.totalStudents.toString(),
        color: AppConstants.primaryColor,
        icon: Icons.people,
      ),
      _StatData(
        label: 'Embeddings',
        value: stats.totalEmbeddings.toString(),
        color: AppConstants.accentColor,
        icon: Icons.memory,
      ),
      _StatData(
        label: 'Records',
        value: stats.totalAttendanceRecords.toString(),
        color: AppConstants.successColor,
        icon: Icons.check_circle,
      ),
      _StatData(
        label: 'Avg Attendance',
        value: '${stats.averageAttendance.toStringAsFixed(1)}%',
        color: AppConstants.warningColor,
        icon: Icons.trending_up,
      ),
    ];
    return Wrap(
      spacing: AppConstants.paddingMedium,
      runSpacing: AppConstants.paddingMedium,
      children: statItems.map((stat) => _statCard(stat)).toList(),
    );
  }

  Widget _statCard(_StatData stat) {
    return Container(
      width: 160,
      decoration: BoxDecoration(
        color: AppConstants.cardColor,
        border: Border.all(color: AppConstants.cardBorder),
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
        boxShadow: [AppConstants.cardShadow],
      ),
      child: Padding(
        padding: const EdgeInsets.all(AppConstants.paddingMedium),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: stat.color.withAlpha(26),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(stat.icon, color: stat.color, size: 20),
            ),
            const SizedBox(height: AppConstants.paddingSmall),
            Text(
              stat.label,
              style: const TextStyle(
                color: AppConstants.textTertiary,
                fontSize: 12,
              ),
            ),
            const SizedBox(height: AppConstants.paddingSmall / 2),
            Text(
              stat.value,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: stat.color,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _studentTile(AttendanceDetails details) {
    final ratio = details.totalClasses > 0
        ? '${details.presentCount}/${details.totalClasses}'
        : '0/0';
    return Card(
      child: InkWell(
        onTap: () => _showStudentDetails(details),
        child: Padding(
          padding: const EdgeInsets.all(AppConstants.paddingMedium),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      details.student.name,
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                    const SizedBox(height: AppConstants.paddingSmall / 2),
                    Text(
                      'Roll: ${details.student.rollNumber} · Class: ${details.student.className}',
                    ),
                    const SizedBox(height: AppConstants.paddingSmall / 2),
                    Text(
                      'Attendees: ${details.presentCount} · Absentees: ${details.absentCount}',
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    ratio,
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: AppConstants.paddingSmall / 2),
                  Text(
                    '${details.attendancePercentage.toStringAsFixed(1)}%',
                    style: const TextStyle(color: Colors.grey),
                  ),
                  const SizedBox(height: AppConstants.paddingSmall / 2),
                  const Icon(Icons.chevron_right, size: 20),
                ],
              ),
              const SizedBox(width: AppConstants.paddingSmall),
              IconButton(
                icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                tooltip: 'Delete student',
                onPressed: () => _handleDeleteStudent(details),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _handleDeleteStudent(AttendanceDetails details) async {
    final student = details.student;
    if (student.id == null) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Student'),
        content: Text(
          'Delete ${student.name}, their embeddings, and all attendance data?\nThis cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    try {
      // Delete embeddings first, then the student
      await _dbManager.deleteEmbeddingsForStudent(student.id!);
      await _dbManager.deleteStudent(student.id!);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('${student.name} deleted'),
          backgroundColor: AppConstants.successColor,
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to delete: $e'),
          backgroundColor: AppConstants.errorColor,
        ),
      );
    } finally {
      _reloadData();
    }
  }

  Widget _attendanceHistoryTile(_DateAttendanceSummary summary) {
    final dateLabel =
        '${summary.date.day.toString().padLeft(2, '0')}/${summary.date.month.toString().padLeft(2, '0')}/${summary.date.year}';
    final percentage = summary.totalStudents > 0
        ? (summary.presentCount / summary.totalStudents * 100)
        : 0.0;
    final percentColor = percentage >= 75
        ? AppConstants.successColor
        : percentage >= 50
        ? AppConstants.warningColor
        : AppConstants.errorColor;

    return Card(
      child: InkWell(
        onTap: () => _showDateDetails(summary),
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
        child: Padding(
          padding: const EdgeInsets.all(AppConstants.paddingMedium),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: percentColor.withAlpha(26),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(
                  Icons.calendar_today,
                  color: percentColor,
                  size: 20,
                ),
              ),
              const SizedBox(width: AppConstants.paddingMedium),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      dateLabel,
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Attendees = ${summary.presentCount}, Absentees = ${summary.absentCount}',
                      style: const TextStyle(
                        fontSize: 12,
                        color: AppConstants.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '${percentage.toStringAsFixed(0)}%',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                      color: percentColor,
                    ),
                  ),
                  Text(
                    '${summary.presentCount}/${summary.totalStudents}',
                    style: const TextStyle(
                      fontSize: 11,
                      color: AppConstants.textTertiary,
                    ),
                  ),
                ],
              ),
              const SizedBox(width: 4),
              const Icon(
                Icons.chevron_right,
                size: 20,
                color: AppConstants.textTertiary,
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showDateDetails(_DateAttendanceSummary summary) {
    final dateLabel =
        '${summary.date.day.toString().padLeft(2, '0')}/${summary.date.month.toString().padLeft(2, '0')}/${summary.date.year}';
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Attendance - $dateLabel'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              _dialogRow('Total Students', summary.totalStudents.toString()),
              _dialogRow('Attendees', summary.presentCount.toString()),
              _dialogRow('Absentees', summary.absentCount.toString()),
              const Divider(),
              const Text(
                'Students',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ...summary.entries.map(
                (e) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 3),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Expanded(
                        child: Text(
                          e.name,
                          style: const TextStyle(fontSize: 13),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      Chip(
                        label: Text(
                          e.status.displayName,
                          style: const TextStyle(fontSize: 11),
                        ),
                        backgroundColor: _statusColor(e.status),
                        visualDensity: VisualDensity.compact,
                        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _reloadData() {
    if (!mounted) return;
    setState(() {
      _systemStatsFuture = _attendanceModule.getSystemStatistics();
      _studentDetailsFuture = _loadStudentDetails();
      _enrolledStudentsFuture = _loadEnrolledStudents();
      _attendanceHistoryFuture = _loadAttendanceHistory();
    });
  }

  Future<List<AttendanceDetails>> _loadStudentDetails() async {
    try {
      final students = await _dbManager.getAllStudents();

      final records = <AttendanceDetails>[];
      for (final student in students) {
        try {
          final detail = await _attendanceModule.getAttendanceDetails(
            student.id!,
          );
          if (detail != null) {
            records.add(detail);
          } else {
            records.add(
              AttendanceDetails(
                student: student,
                totalClasses: 0,
                presentCount: 0,
                absentCount: 0,
                lateCount: 0,
                attendancePercentage: 0.0,
                records: [],
              ),
            );
          }
        } catch (e) {
          records.add(
            AttendanceDetails(
              student: student,
              totalClasses: 0,
              presentCount: 0,
              absentCount: 0,
              lateCount: 0,
              attendancePercentage: 0.0,
              records: [],
            ),
          );
        }
      }
      records.sort((a, b) => a.student.name.compareTo(b.student.name));
      return records;
    } catch (e) {
      return [];
    }
  }

  Future<List<AttendanceDetails>> _loadEnrolledStudents() async {
    try {
      final enrolledStudents = await _dbManager.getEnrolledStudents();

      final records = <AttendanceDetails>[];
      for (final student in enrolledStudents) {
        try {
          final detail = await _attendanceModule.getAttendanceDetails(
            student.id!,
          );
          if (detail != null) {
            records.add(detail);
          } else {
            records.add(
              AttendanceDetails(
                student: student,
                totalClasses: 0,
                presentCount: 0,
                absentCount: 0,
                lateCount: 0,
                attendancePercentage: 0.0,
                records: [],
              ),
            );
          }
        } catch (e) {
          records.add(
            AttendanceDetails(
              student: student,
              totalClasses: 0,
              presentCount: 0,
              absentCount: 0,
              lateCount: 0,
              attendancePercentage: 0.0,
              records: [],
            ),
          );
        }
      }
      records.sort((a, b) => a.student.name.compareTo(b.student.name));
      return records;
    } catch (e) {
      return [];
    }
  }

  Future<List<_DateAttendanceSummary>> _loadAttendanceHistory() async {
    try {
      final allRecords = await _dbManager.getAllAttendance();
      if (allRecords.isEmpty) return [];

      // Group records by date, deduplicate per student per date (latest wins)
      final byDate = <String, Map<int, AttendanceRecord>>{};
      for (final record in allRecords) {
        final dateKey =
            '${record.date.year}-${record.date.month.toString().padLeft(2, '0')}-${record.date.day.toString().padLeft(2, '0')}';
        byDate.putIfAbsent(dateKey, () => {});
        final existing = byDate[dateKey]![record.studentId];
        if (existing == null ||
            record.recordedAt.isAfter(existing.recordedAt)) {
          byDate[dateKey]![record.studentId] = record;
        }
      }

      final summaries = <_DateAttendanceSummary>[];
      for (final entry in byDate.entries) {
        final records = entry.value.values.toList();
        final date = records.first.date;
        final present = records
            .where((r) => r.status == AttendanceStatus.present)
            .length;
        final absent = records
            .where((r) => r.status == AttendanceStatus.absent)
            .length;
        final late = records
            .where((r) => r.status == AttendanceStatus.late)
            .length;

        // Get student names for the detail view
        final entries = <_DateStudentEntry>[];
        for (final record in records) {
          final student = await _dbManager.getStudentById(record.studentId);
          entries.add(
            _DateStudentEntry(
              name: student?.name ?? 'Unknown',
              status: record.status,
            ),
          );
        }
        entries.sort((a, b) => a.name.compareTo(b.name));

        summaries.add(
          _DateAttendanceSummary(
            date: DateTime(date.year, date.month, date.day),
            totalStudents: records.length,
            presentCount: present,
            absentCount: absent,
            lateCount: late,
            entries: entries,
          ),
        );
      }

      // Sort by date descending (most recent first)
      summaries.sort((a, b) => b.date.compareTo(a.date));
      return summaries;
    } catch (e) {
      return [];
    }
  }

  void _showStudentDetails(AttendanceDetails details) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(details.student.name),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _dialogRow('Roll', details.student.rollNumber),
              _dialogRow('Class', details.student.className),
              const SizedBox(height: AppConstants.paddingMedium),
              _dialogRow('Total Classes', details.totalClasses.toString()),
              _dialogRow('Present', details.presentCount.toString()),
              _dialogRow('Absent', details.absentCount.toString()),
              _dialogRow('Late', details.lateCount.toString()),
              _dialogRow(
                'Attendance %',
                '${details.attendancePercentage.toStringAsFixed(1)}%',
              ),
              const SizedBox(height: AppConstants.paddingMedium),
              const Text(
                'Full record',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: AppConstants.paddingSmall),
              ...details.records.map(_recordRow),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  Widget _dialogRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        vertical: AppConstants.paddingSmall / 2,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.grey)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  Widget _recordRow(AttendanceRecord record) {
    final dateLabel =
        '${record.date.day.toString().padLeft(2, '0')}-${record.date.month.toString().padLeft(2, '0')}-${record.date.year % 100}';
    final timeLabel = (record.time?.isNotEmpty ?? false)
        ? ' (${record.time})'
        : '';
    return Padding(
      padding: const EdgeInsets.symmetric(
        vertical: AppConstants.paddingSmall / 2,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('$dateLabel$timeLabel'),
          Chip(
            label: Text(record.status.displayName),
            backgroundColor: _statusColor(record.status),
          ),
        ],
      ),
    );
  }

  Color _statusColor(AttendanceStatus status) {
    switch (status) {
      case AttendanceStatus.present:
        return AppConstants.successColor.withValues(alpha: 0.15);
      case AttendanceStatus.absent:
        return AppConstants.errorColor.withValues(alpha: 0.15);
      case AttendanceStatus.late:
        return AppConstants.warningColor.withValues(alpha: 0.15);
    }
  }
}

class _StatData {
  final String label;
  final String value;
  final Color color;
  final IconData icon;

  _StatData({
    required this.label,
    required this.value,
    required this.color,
    required this.icon,
  });
}

class _DateAttendanceSummary {
  final DateTime date;
  final int totalStudents;
  final int presentCount;
  final int absentCount;
  final int lateCount;
  final List<_DateStudentEntry> entries;

  _DateAttendanceSummary({
    required this.date,
    required this.totalStudents,
    required this.presentCount,
    required this.absentCount,
    required this.lateCount,
    required this.entries,
  });
}

class _DateStudentEntry {
  final String name;
  final AttendanceStatus status;

  _DateStudentEntry({required this.name, required this.status});
}


