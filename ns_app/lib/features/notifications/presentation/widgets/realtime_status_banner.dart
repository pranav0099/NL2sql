import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../../core/realtime/stomp_service.dart';
import '../../../../core/realtime/stomp_state.dart';

/// A subtle connection status banner (like WhatsApp's
/// "Connecting..." / "Connected" indicator).
///
/// Shows at the top of the screen:
/// - **connecting** → amber bar "Connecting to live updates..."
/// - **error** → red bar "Live updates unavailable. Tap to retry."
/// - **connected** → green bar "Connected" (auto-hides after 2 seconds)
/// - **disconnected** → hidden
class RealtimeStatusBanner extends ConsumerStatefulWidget {
  const RealtimeStatusBanner({super.key});

  @override
  ConsumerState<RealtimeStatusBanner> createState() =>
      _RealtimeStatusBannerState();
}

class _RealtimeStatusBannerState extends ConsumerState<RealtimeStatusBanner> {
  bool _showConnected = false;
  Timer? _hideTimer;

  @override
  void dispose() {
    _hideTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final stompState = ref.watch(stompStateProvider);

    return stompState.when(
      data: (state) => _buildBanner(state),
      loading: () => const SizedBox.shrink(),
      error: (err, stack) => const SizedBox.shrink(),
    );
  }

  Widget _buildBanner(StompState state) {
    return state.when(
      disconnected: () => const SizedBox.shrink(),
      connecting: () => _banner(
        color: const Color(0xFFF59E0B),
        icon: Icons.sync_rounded,
        text: 'Connecting to live updates...',
        spinning: true,
      ),
      connected: (_) {
        // Auto-hide after 2 seconds.
        if (!_showConnected) {
          _showConnected = true;
          _hideTimer?.cancel();
          _hideTimer = Timer(const Duration(seconds: 2), () {
            if (mounted) setState(() => _showConnected = false);
          });
        }
        if (!_showConnected) return const SizedBox.shrink();
        return _banner(
          color: const Color(0xFF16A34A),
          icon: Icons.wifi_rounded,
          text: 'Connected',
        );
      },
      error: (message) => _banner(
        color: const Color(0xFFEF4444),
        icon: Icons.wifi_off_rounded,
        text: 'Live updates unavailable. Tap to retry.',
        onTap: () => ref.read(stompServiceProvider).retry(),
      ),
    );
  }

  Widget _banner({
    required Color color,
    required IconData icon,
    required String text,
    bool spinning = false,
    VoidCallback? onTap,
  }) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOut,
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        border: Border(
          bottom: BorderSide(color: color.withValues(alpha: 0.2)),
        ),
      ),
      child: GestureDetector(
        onTap: onTap,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (spinning)
              SizedBox(
                width: 14,
                height: 14,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation(color),
                ),
              )
            else
              Icon(icon, size: 14, color: color),
            const SizedBox(width: 8),
            Text(
              text,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
