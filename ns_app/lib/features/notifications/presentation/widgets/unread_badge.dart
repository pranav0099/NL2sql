import 'package:flutter/material.dart';

/// A red dot badge that overlays a child widget (typically an icon)
/// to indicate unread notification count.
///
/// Hides when [count] is 0. Displays "9+" for counts above 9.
/// Animates in/out with a [ScaleTransition].
class UnreadBadge extends StatelessWidget {
  final Widget child;
  final int count;
  final double size;
  final Color badgeColor;

  const UnreadBadge({
    super.key,
    required this.child,
    required this.count,
    this.size = 18.0,
    this.badgeColor = const Color(0xFFEF4444),
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        child,
        Positioned(
          right: -4,
          top: -4,
          child: AnimatedScale(
            scale: count > 0 ? 1.0 : 0.0,
            duration: const Duration(milliseconds: 200),
            curve: Curves.easeOutBack,
            child: Container(
              constraints: BoxConstraints(
                minWidth: size,
                minHeight: size,
              ),
              padding: const EdgeInsets.symmetric(horizontal: 4),
              decoration: BoxDecoration(
                color: badgeColor,
                borderRadius: BorderRadius.circular(size / 2),
                boxShadow: [
                  BoxShadow(
                    color: badgeColor.withValues(alpha: 0.4),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              alignment: Alignment.center,
              child: Text(
                count > 9 ? '9+' : count.toString(),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  height: 1,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ),
      ],
    );
  }
}
