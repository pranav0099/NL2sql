import 'dart:async';
import 'package:flutter/material.dart';
import '../../../../core/realtime/stock_update_message.dart';

/// Animated badge that shows the current stock level.
///
/// Uses [AnimatedSwitcher] so stock transitions are smooth (fade + scale).
/// Shows a brief "Updated just now" text for 3 seconds then fades.
/// Wrapped in a [SizedBox] with consistent width to prevent layout jumps.
class StockBadge extends StatefulWidget {
  final StockLevel stockLevel;
  final bool showUpdatedText;

  const StockBadge({
    super.key,
    required this.stockLevel,
    this.showUpdatedText = false,
  });

  @override
  State<StockBadge> createState() => _StockBadgeState();
}

class _StockBadgeState extends State<StockBadge> {
  bool _showUpdated = false;
  Timer? _hideTimer;

  @override
  void didUpdateWidget(StockBadge oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.stockLevel != widget.stockLevel) {
      if (widget.showUpdatedText) {
        setState(() => _showUpdated = true);
        _hideTimer?.cancel();
        _hideTimer = Timer(const Duration(seconds: 3), () {
          if (mounted) setState(() => _showUpdated = false);
        });
      }
    }
  }

  @override
  void dispose() {
    _hideTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: 100,
          height: 28,
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            switchInCurve: Curves.easeOutBack,
            switchOutCurve: Curves.easeIn,
            transitionBuilder: (child, animation) {
              return FadeTransition(
                opacity: animation,
                child: ScaleTransition(
                  scale: animation,
                  child: child,
                ),
              );
            },
            child: _BadgeChip(
              key: ValueKey(widget.stockLevel),
              stockLevel: widget.stockLevel,
            ),
          ),
        ),
        AnimatedOpacity(
          opacity: _showUpdated ? 1.0 : 0.0,
          duration: const Duration(milliseconds: 300),
          child: Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              'Updated just now',
              style: TextStyle(
                fontSize: 10,
                color: _colorForLevel(widget.stockLevel).withValues(alpha: 0.7),
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ),
      ],
    );
  }

  Color _colorForLevel(StockLevel level) {
    switch (level) {
      case StockLevel.inStock:
        return const Color(0xFF16A34A);
      case StockLevel.lowStock:
        return const Color(0xFFF59E0B);
      case StockLevel.outOfStock:
        return const Color(0xFFEF4444);
    }
  }
}

class _BadgeChip extends StatelessWidget {
  final StockLevel stockLevel;

  const _BadgeChip({super.key, required this.stockLevel});

  @override
  Widget build(BuildContext context) {
    final color = _colorForLevel(stockLevel);
    final label = _labelForLevel(stockLevel);
    final icon = _iconForLevel(stockLevel);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Color _colorForLevel(StockLevel level) {
    switch (level) {
      case StockLevel.inStock:
        return const Color(0xFF16A34A);
      case StockLevel.lowStock:
        return const Color(0xFFF59E0B);
      case StockLevel.outOfStock:
        return const Color(0xFFEF4444);
    }
  }

  String _labelForLevel(StockLevel level) {
    switch (level) {
      case StockLevel.inStock:
        return 'In Stock';
      case StockLevel.lowStock:
        return 'Low Stock';
      case StockLevel.outOfStock:
        return 'Out of Stock';
    }
  }

  IconData _iconForLevel(StockLevel level) {
    switch (level) {
      case StockLevel.inStock:
        return Icons.check_circle_outline_rounded;
      case StockLevel.lowStock:
        return Icons.warning_amber_rounded;
      case StockLevel.outOfStock:
        return Icons.cancel_outlined;
    }
  }
}
