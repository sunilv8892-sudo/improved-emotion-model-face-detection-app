import 'package:flutter/material.dart';
import '../utils/constants.dart';
import '../widgets/animated_background.dart';
import '../database/database_manager.dart';
import '../modules/m4_attendance_management.dart';
import '../models/attendance_model.dart';

/// Home Screen (Page 1)
/// Main navigation hub with buttons to all features
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late final DatabaseManager _dbManager;
  late final AttendanceManagementModule _attendanceModule;

  int _totalStudents = 0;
  int _presentToday = 0;
  int _totalSessions = 0;

  @override
  void initState() {
    super.initState();
    _dbManager = DatabaseManager();
    _attendanceModule = AttendanceManagementModule(_dbManager);
    _loadStats();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Reload stats when screen comes into focus
    _loadStats();
  }

  Future<void> _loadStats() async {
    try {
      // Get total students count
      final students = await _dbManager.getAllStudents();
      final totalStudents = students.length;

      // Get today's attendance count (unique students marked present)
      final today = DateTime.now();
      final todayRecords = await _dbManager.getAttendanceForDate(today);
      
      // Count unique students who are marked as present today
      final presentStudentIds = todayRecords
          .where((record) => record.status == AttendanceStatus.present)
          .map((record) => record.studentId)
          .toSet(); // Use Set to get unique student IDs
      
      final presentToday = presentStudentIds.length;

      // Get total sessions (unique dates with attendance records)
      final allAttendance = await _dbManager.getAllAttendance();
      final uniqueDates = allAttendance
          .map(
            (record) =>
                DateTime(record.date.year, record.date.month, record.date.day),
          )
          .toSet()
          .length;

      setState(() {
        _totalStudents = totalStudents;
        _presentToday = presentToday;
        _totalSessions = uniqueDates;
      });
    } catch (e) {
      // Handle errors gracefully
      setState(() {
        _totalStudents = 0;
        _presentToday = 0;
        _totalSessions = 0;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppConstants.backgroundColor,
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.transparent,
        titleSpacing: 0,
        title: Padding(
          padding: const EdgeInsets.only(left: 20),
          child: ShaderMask(
            shaderCallback: (bounds) => const LinearGradient(
              colors: [Color(0xFF00D4FF), Color(0xFF6C63FF)],
            ).createShader(bounds),
            child: const Text(
              'FaceAttend',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.w800,
                color: Colors.white,
                letterSpacing: 0.5,
              ),
            ),
          ),
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.white.withValues(alpha: 0.15)),
              ),
              child: IconButton(
                icon: const Icon(Icons.info_outline_rounded, color: Color(0xFF00D4FF), size: 22),
                onPressed: () => _showAboutDialog(context),
              ),
            ),
          ),
        ],
      ),
      body: AnimatedBackground(
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildHeaderSection(),
                const SizedBox(height: 16),
                _buildStatsRow(),
                const SizedBox(height: 24),
                _buildSectionLabel('Featured', const Color(0xFF00D4FF)),
                const SizedBox(height: 12),
                _buildFeaturedRow(context),
                const SizedBox(height: 24),
                _buildSectionLabel('More Tools', const Color(0xFFFFB830)),
                const SizedBox(height: 12),
                _buildToolsGrid(context),
                const SizedBox(height: 12),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ────────── Section label ──────────
  Widget _buildSectionLabel(String text, Color accentDot) {
    return Row(
      children: [
        Container(
          width: 4, height: 18,
          decoration: BoxDecoration(
            color: accentDot,
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        const SizedBox(width: 8),
        Text(
          text,
          style: const TextStyle(
            color: Color(0xFFFFFFFF),
            fontSize: 16,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.3,
          ),
        ),
      ],
    );
  }

  // ────────── Stats Row ──────────
  Widget _buildStatsRow() {
    return Row(
      children: [
        _buildStatChip(Icons.groups_rounded, '$_totalStudents', 'Students', const Color(0xFF6C63FF), const Color(0xFF9B59F5)),
        const SizedBox(width: 10),
        _buildStatChip(Icons.how_to_reg_rounded, '$_presentToday', 'Present', const Color(0xFF00E096), const Color(0xFF00A878)),
        const SizedBox(width: 10),
        _buildStatChip(Icons.calendar_month_rounded, '$_totalSessions', 'Sessions', const Color(0xFFFFB830), const Color(0xFFFF7043)),
      ],
    );
  }

  Widget _buildStatChip(IconData icon, String value, String label, Color c1, Color c2) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [c1.withValues(alpha: 0.22), c2.withValues(alpha: 0.12)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: c1.withValues(alpha: 0.45), width: 1.2),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: c1, size: 22),
            const SizedBox(height: 6),
            Text(
              value,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w800,
                color: c1,
              ),
            ),
            Text(
              label,
              style: const TextStyle(
                fontSize: 10,
                color: Color(0xFFCDD5E0),
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ────────── Featured Row (Enroll + Attendance) ──────────
  Widget _buildFeaturedRow(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: _buildFeaturedCard(
            context,
            icon: Icons.person_add_alt_1_rounded,
            title: 'Enroll',
            subtitle: 'Add new student faces',
            gradient: const LinearGradient(
              colors: [Color(0xFF6C63FF), Color(0xFF9B59F5)],
              begin: Alignment.topLeft, end: Alignment.bottomRight,
            ),
            route: AppConstants.routeEnroll,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildFeaturedCard(
            context,
            icon: Icons.face_retouching_natural,
            title: 'Attendance',
            subtitle: 'Scan & mark faces',
            gradient: const LinearGradient(
              colors: [Color(0xFF00D4FF), Color(0xFF00A878)],
              begin: Alignment.topLeft, end: Alignment.bottomRight,
            ),
            route: AppConstants.routeAttendance,
          ),
        ),
      ],
    );
  }

  Widget _buildFeaturedCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String subtitle,
    required LinearGradient gradient,
    required String route,
  }) {
    return GestureDetector(
      onTap: () => Navigator.pushNamed(context, route),
      child: Container(
        height: 148,
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(24),
          boxShadow: [
            BoxShadow(
              color: gradient.colors.first.withValues(alpha: 0.45),
              blurRadius: 22,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: Colors.white, size: 26),
            ),
            const Spacer(),
            Text(
              title,
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w800,
                color: Colors.white,
                letterSpacing: 0.2,
              ),
            ),
            const SizedBox(height: 3),
            Text(
              subtitle,
              style: TextStyle(
                fontSize: 11,
                color: Colors.white.withValues(alpha: 0.82),
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ────────── Tools Grid (4 items) ──────────
  Widget _buildToolsGrid(BuildContext context) {
    final tools = [
      _ToolItem(Icons.mood_rounded, 'Expression', 'Emotions AI', const Color(0xFFFFB830), const Color(0xFFFF7043), AppConstants.routeExpressionDetection),
      _ToolItem(Icons.download_rounded, 'Export', 'Reports', const Color(0xFF00D4FF), const Color(0xFF00A8E8), AppConstants.routeExport),
      _ToolItem(Icons.settings_rounded, 'Settings', 'Configure', const Color(0xFF6C63FF), const Color(0xFF9B59F5), AppConstants.routeSettings),
      _ToolItem(Icons.storage_rounded, 'Database', 'Manage data', const Color(0xFF00E096), const Color(0xFF00A878), AppConstants.routeDatabase),
    ];
    return GridView.count(
      crossAxisCount: 2,
      mainAxisSpacing: 12,
      crossAxisSpacing: 12,
      childAspectRatio: 1.55,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      children: tools.map((t) => _buildToolCard2(context, t)).toList(),
    );
  }

  Widget _buildToolCard2(BuildContext context, _ToolItem t) {
    return GestureDetector(
      onTap: () => Navigator.pushNamed(context, t.route),
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFF1B2A49),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: t.c1.withValues(alpha: 0.35),
            width: 1.2,
          ),
          boxShadow: [
            BoxShadow(
              color: t.c1.withValues(alpha: 0.18),
              blurRadius: 16,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [t.c1, t.c2],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: t.c1.withValues(alpha: 0.4),
                    blurRadius: 10,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: Icon(t.icon, color: Colors.white, size: 20),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    t.title,
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    t.subtitle,
                    style: const TextStyle(
                      fontSize: 10,
                      color: Color(0xFF8B9BB4),
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right_rounded, color: t.c1, size: 18),
          ],
        ),
      ),
    );
  }

  Widget _buildHeaderSection() {
    final now = DateTime.now();
    final greeting = now.hour < 12 ? 'Good Morning' : now.hour < 17 ? 'Good Afternoon' : 'Good Evening';
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF1B2A49), Color(0xFF243354)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withValues(alpha: 0.1), width: 1),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF6C63FF).withValues(alpha: 0.2),
            blurRadius: 24,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  greeting,
                  style: const TextStyle(
                    fontSize: 13,
                    color: Color(0xFF00D4FF),
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.3,
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'AI Face Attendance',
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                    letterSpacing: 0.2,
                  ),
                ),
                const SizedBox(height: 6),
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: const Color(0xFF00E096).withValues(alpha: 0.18),
                        borderRadius: BorderRadius.circular(999),
                        border: Border.all(color: const Color(0xFF00E096).withValues(alpha: 0.5)),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.circle, color: Color(0xFF00E096), size: 7),
                          SizedBox(width: 4),
                          Text('Offline Ready', style: TextStyle(fontSize: 10, color: Color(0xFF00E096), fontWeight: FontWeight.w600)),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFF6C63FF), Color(0xFF00D4FF)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(18),
              boxShadow: [
                BoxShadow(
                  color: const Color(0xFF6C63FF).withValues(alpha: 0.5),
                  blurRadius: 20,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: const Icon(Icons.face_retouching_natural, color: Colors.white, size: 32),
          ),
        ],
      ),
    );
  }

  // _buildDashboardGrid replaced by _buildFeaturedRow + _buildToolsGrid above

  Widget _buildQuickToolsSection(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        horizontal: AppConstants.paddingMedium,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(
              left: AppConstants.paddingSmall,
              bottom: AppConstants.paddingMedium,
            ),
            child: Text(
              'Management Tools',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: AppConstants.textPrimary,
              ),
            ),
          ),
          Row(
            children: [
              Expanded(
                child: _buildToolCard(
                  context,
                  icon: Icons.storage,
                  title: 'Database',
                  subtitle: 'View & manage students',
                  route: AppConstants.routeDatabase,
                ),
              ),
              const SizedBox(width: AppConstants.paddingMedium),
              Expanded(
                child: _buildToolCard(
                  context,
                  icon: Icons.download_rounded,
                  title: 'Export',
                  subtitle: 'Generate reports',
                  route: AppConstants.routeExport,
                ),
              ),
              const SizedBox(width: AppConstants.paddingMedium),
              Expanded(
                child: _buildToolCard(
                  context,
                  icon: Icons.tune,
                  title: 'Settings',
                  subtitle: 'Configure app',
                  route: AppConstants.routeSettings,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildFeaturesPreview() {
    return Container(
      margin: const EdgeInsets.symmetric(
        horizontal: AppConstants.paddingMedium,
      ),
      padding: const EdgeInsets.all(AppConstants.paddingLarge),
      decoration: BoxDecoration(
        color: AppConstants.cardColor,
        borderRadius: BorderRadius.circular(AppConstants.borderRadiusLarge),
        border: Border.all(color: AppConstants.cardBorder),
        boxShadow: [AppConstants.cardShadow],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.star, color: AppConstants.primaryColor, size: 20),
              const SizedBox(width: 8),
              const Text(
                'Key Features',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: AppConstants.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: AppConstants.paddingMedium),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              _buildFeatureItem(Icons.offline_bolt, 'Offline Operation'),
              _buildFeatureItem(Icons.speed, 'Real-time Detection'),
              _buildFeatureItem(Icons.memory, 'Smart Embeddings'),
              _buildFeatureItem(Icons.verified, 'Accurate Matching'),
              _buildFeatureItem(Icons.history, 'Attendance Logs'),
              _buildFeatureItem(Icons.file_download, 'Export Reports'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildModernActionCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String subtitle,
    required String route,
    required Gradient gradient,
  }) {
    return InkWell(
      onTap: () => Navigator.pushNamed(context, route),
      borderRadius: BorderRadius.circular(AppConstants.borderRadiusLarge),
      child: Container(
        height: 140,
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(AppConstants.borderRadiusLarge),
          boxShadow: [
            BoxShadow(
              color: gradient.colors.first.withAlpha(100),
              blurRadius: 15,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Padding(
          padding: const EdgeInsets.all(AppConstants.paddingLarge),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white.withAlpha(30),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: Colors.white, size: 28),
              ),
              const Spacer(),
              Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                subtitle,
                style: TextStyle(
                  color: Colors.white.withAlpha(220),
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildToolCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String subtitle,
    required String route,
  }) {
    return InkWell(
      onTap: () => Navigator.pushNamed(context, route),
      borderRadius: BorderRadius.circular(AppConstants.borderRadiusLarge),
      child: Container(
        height: 100,
        decoration: BoxDecoration(
          color: AppConstants.cardColor,
          borderRadius: BorderRadius.circular(AppConstants.borderRadiusLarge),
          border: Border.all(color: AppConstants.cardBorder),
          boxShadow: [AppConstants.cardShadow],
        ),
        child: Padding(
          padding: const EdgeInsets.all(AppConstants.paddingMedium),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppConstants.primaryColor.withAlpha(15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, size: 24, color: AppConstants.primaryColor),
              ),
              const SizedBox(height: AppConstants.paddingSmall),
              Text(
                title,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                  color: AppConstants.textPrimary,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                subtitle,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 11,
                  color: AppConstants.textTertiary,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFeatureItem(IconData icon, String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppConstants.inputFill,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppConstants.cardBorder.withAlpha(100)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: AppConstants.primaryColor),
          const SizedBox(width: 6),
          Text(
            label,
            style: const TextStyle(
              fontSize: 12,
              color: AppConstants.textSecondary,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  // _buildBottomStatsBar / _buildStatItem replaced by _buildStatsRow / _buildStatChip above

  // _buildHexagonalCard / _buildCircularCard removed — replaced by _buildFeaturedCard / _buildToolCard2

  Widget _buildFeatureHighlightCard() {
    return Container(
      decoration: BoxDecoration(
        color: AppConstants.cardColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppConstants.primaryColor.withAlpha(30)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.offline_bolt,
              color: AppConstants.primaryColor,
              size: 24,
            ),
            const SizedBox(height: 8),
            const Text(
              'Offline\nOperation',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: AppConstants.textPrimary,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'Works without internet',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 10, color: AppConstants.textSecondary),
            ),
          ],
        ),
      ),
    );
  }

  void _showAboutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('About Face Attendance'),
        content: const Text(
          'AI-powered face recognition system for seamless attendance tracking. '
          'Works completely offline with high accuracy and real-time detection.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}

// ────────────────────────────────────────────────────────────
// Helper data class for tools grid
// ────────────────────────────────────────────────────────────
class _ToolItem {
  const _ToolItem(this.icon, this.title, this.subtitle, this.c1, this.c2, this.route);
  final IconData icon;
  final String title;
  final String subtitle;
  final Color c1;
  final Color c2;
  final String route;
}
