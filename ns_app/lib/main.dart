import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';

// Services
import 'core/realtime/stomp_service.dart';
import 'core/realtime/mock_stomp_service.dart';
import 'core/realtime/stock_update_message.dart';
import 'core/notifications/local_notification_service.dart';
import 'core/notifications/notification_router.dart';
import 'core/notifications/fcm_service.dart';

// Providers & Screens & Widgets
import 'features/notifications/presentation/providers/notification_provider.dart';
import 'features/notifications/presentation/screens/notifications_screen.dart';
import 'features/notifications/presentation/widgets/unread_badge.dart';
import 'features/notifications/presentation/widgets/stock_badge.dart';
import 'features/notifications/presentation/widgets/realtime_status_banner.dart';
import 'features/notifications/domain/models/app_notification.dart';

// Set useMockStomp to true to run fully offline with mock triggers.
const bool useMockStomp = true;

@pragma('vm:entry-point')
Future<void> _firebaseBackgroundHandler(RemoteMessage message) async {
  try {
    await Firebase.initializeApp();
  } catch (e) {
    debugPrint('Firebase background init failed or not configured: $e');
  }
  debugPrint('Handling a background message: ${message.messageId}');
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Firebase messaging & local notifications
  if (!useMockStomp) {
    try {
      await Firebase.initializeApp();
      FirebaseMessaging.onBackgroundMessage(_firebaseBackgroundHandler);
    } catch (e) {
      debugPrint('Firebase initialization skipped/failed: $e. Using mock mode.');
    }
  }

  // Always initialize local notifications
  await LocalNotificationService().init();

  runApp(
    const ProviderScope(
      child: MyApp(),
    ),
  );
}

// Global Provider for routing and navigation state
final navigationShellProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    navigatorKey: NotificationRouter.navigatorKey,
    initialLocation: '/customer/products',
    routes: [
      ShellRoute(
        builder: (context, state, child) {
          return CustomerShell(child: child);
        },
        routes: [
          GoRoute(
            path: '/customer/products',
            builder: (context, state) => const ProductsDemoScreen(),
          ),
          GoRoute(
            path: '/customer/notifications',
            builder: (context, state) => const NotificationsScreen(),
          ),
          GoRoute(
            path: '/customer/retailer/:id',
            builder: (context, state) {
              final retailerId = state.pathParameters['id'] ?? 'retailer_1';
              return RetailerDetailStubScreen(retailerId: retailerId);
            },
          ),
        ],
      ),
    ],
  );
});

class MyApp extends ConsumerWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final router = ref.watch(navigationShellProvider);

    // Initialise FCM & STOMP connection when app starts
    ref.listen(stompServiceProvider, (previous, next) {
      // Connect to Stomp Service
      if (useMockStomp) {
        ref.read(mockStompServiceProvider).connect('mock_token', userId: 'user_123');
        // Register mock trigger listener to update NotificationNotifier
        ref.read(mockStompServiceProvider).onNotificationReceived = (payload) {
          final notification = AppNotification(
            id: DateTime.now().millisecondsSinceEpoch.toString(),
            type: payload['type'] == 'DEAL'
                ? NotificationType.deal
                : NotificationType.stockUpdate,
            title: payload['title'] ?? 'Stock Update',
            body: payload['body'] ?? '',
            retailerId: payload['retailerId'],
            productId: payload['productId'],
            sentAt: DateTime.now(),
            isRead: false,
          );
          ref.read(notificationProvider.notifier).addIncoming(notification);
          LocalNotificationService().showRawNotification(
            title: notification.title,
            body: notification.body,
            data: payload,
          );
        };
      } else {
        next.connect('real_token', userId: 'user_123');
        next.onNotificationReceived = (payload) {
          final notification = AppNotification(
            id: DateTime.now().millisecondsSinceEpoch.toString(),
            type: payload['type'] == 'DEAL'
                ? NotificationType.deal
                : NotificationType.stockUpdate,
            title: payload['title'] ?? 'Stock Update',
            body: payload['body'] ?? '',
            retailerId: payload['retailerId'],
            productId: payload['productId'],
            sentAt: DateTime.now(),
            isRead: false,
          );
          ref.read(notificationProvider.notifier).addIncoming(notification);
        };
      }
    });

    // Start FCM service if not in mock
    if (!useMockStomp) {
      Future.microtask(() {
        ref.read(fcmServiceProvider).init(isAuthenticated: true);
      });
    }

    return MaterialApp.router(
      title: 'NearShop',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1A56DB),
          primary: const Color(0xFF1A56DB),
          secondary: const Color(0xFF1E40AF),
          surface: Colors.white,
        ),
        scaffoldBackgroundColor: const Color(0xFFF9FAFB),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          elevation: 0,
          scrolledUnderElevation: 0,
          iconTheme: IconThemeData(color: Colors.black87),
          titleTextStyle: TextStyle(
            color: Colors.black87,
            fontSize: 20,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
      routerConfig: router,
    );
  }
}

class CustomerShell extends ConsumerWidget {
  final Widget child;

  const CustomerShell({super.key, required this.child});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final unreadCount = ref.watch(unreadCountProvider);
    final router = GoRouter.of(context);
    final currentLocation = router.routerDelegate.currentConfiguration.uri.toString();

    int currentIndex = 0;
    if (currentLocation.startsWith('/customer/notifications')) {
      currentIndex = 1;
    }

    return Scaffold(
      body: child,
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.05),
              blurRadius: 10,
              offset: const Offset(0, -2),
            ),
          ],
        ),
        child: BottomNavigationBar(
          currentIndex: currentIndex,
          onTap: (index) {
            if (index == 0) {
              context.go('/customer/products');
            } else {
              context.go('/customer/notifications');
            }
          },
          selectedItemColor: const Color(0xFF1A56DB),
          unselectedItemColor: Colors.grey.shade500,
          selectedLabelStyle: const TextStyle(fontWeight: FontWeight.w600),
          backgroundColor: Colors.white,
          elevation: 0,
          type: BottomNavigationBarType.fixed,
          items: [
            const BottomNavigationBarItem(
              icon: Icon(Icons.search_rounded),
              activeIcon: Icon(Icons.search_rounded),
              label: 'Search',
            ),
            BottomNavigationBarItem(
              icon: UnreadBadge(
                count: unreadCount,
                child: const Icon(Icons.notifications_outlined),
              ),
              activeIcon: UnreadBadge(
                count: unreadCount,
                child: const Icon(Icons.notifications),
              ),
              label: 'Notifications',
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Demo Screens
// ---------------------------------------------------------------------------

class ProductsDemoScreen extends ConsumerStatefulWidget {
  const ProductsDemoScreen({super.key});

  @override
  ConsumerState<ProductsDemoScreen> createState() => _ProductsDemoScreenState();
}

class _ProductsDemoScreenState extends ConsumerState<ProductsDemoScreen> {
  // Local state for stock levels to showcase dynamic changes in stock badges
  final Map<String, StockLevel> _stockLevels = {
    'product_101': StockLevel.inStock,
    'product_102': StockLevel.lowStock,
    'product_103': StockLevel.outOfStock,
  };

  @override
  void initState() {
    super.initState();
    // In mock mode, let's subscribe to mock STOMP service stock changes
    if (useMockStomp) {
      ref.read(mockStompServiceProvider).subscribeToRetailer('retailer_1', (msg) {
        if (mounted) {
          setState(() {
            _stockLevels[msg.productId] = msg.stockLevel;
          });
        }
      });
    }
  }

  void _simulateStockChange(String productId, String productName, StockLevel level) {
    if (useMockStomp) {
      ref.read(mockStompServiceProvider).triggerMockStockUpdate(
            retailerId: 'retailer_1',
            productId: productId,
            productName: productName,
            stockLevel: level,
          );
    }
  }

  void _triggerPromoDeal() {
    if (useMockStomp) {
      ref.read(mockStompServiceProvider).triggerMockNotification(
            title: 'Special Deal at Tech Store!',
            body: 'Get 20% off on all items using code TECH20!',
            type: 'DEAL',
            retailerId: 'retailer_1',
          );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Discover Products'),
      ),
      body: Column(
        children: [
          const RealtimeStatusBanner(),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(16),
              children: [
                _buildSimulationControls(theme),
                const SizedBox(height: 24),
                Text(
                  'Hyperlocal Inventory',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
                const SizedBox(height: 12),
                _buildProductCard(
                  theme,
                  id: 'product_101',
                  name: 'iPhone 15 Pro',
                  retailer: 'Tech Store',
                  image: '📱',
                  price: r'$999',
                ),
                const SizedBox(height: 12),
                _buildProductCard(
                  theme,
                  id: 'product_102',
                  name: 'Samsung S24',
                  retailer: 'Mobile Hub',
                  image: '📲',
                  price: r'$899',
                ),
                const SizedBox(height: 12),
                _buildProductCard(
                  theme,
                  id: 'product_103',
                  name: 'MacBook Pro',
                  retailer: 'Tech Store',
                  image: '💻',
                  price: r'$1999',
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSimulationControls(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A56DB).withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF1A56DB).withValues(alpha: 0.15)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.bolt, color: Color(0xFF1A56DB)),
              const SizedBox(width: 8),
              Text(
                'Phase 4 Simulation Control Panel',
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: const Color(0xFF1A56DB),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          const Text(
            'Tap buttons below to dispatch mock WebSockets and push notifications instantly to the UI:',
            style: TextStyle(fontSize: 12, color: Colors.black54),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              ActionChip(
                backgroundColor: Colors.white,
                label: const Text('Update iPhone Stock'),
                onPressed: () {
                  final levels = [StockLevel.inStock, StockLevel.lowStock, StockLevel.outOfStock];
                  final current = _stockLevels['product_101'] ?? StockLevel.inStock;
                  final next = levels[(levels.indexOf(current) + 1) % levels.length];
                  _simulateStockChange('product_101', 'iPhone 15 Pro', next);
                },
              ),
              ActionChip(
                backgroundColor: Colors.white,
                label: const Text('Update MacBook Stock'),
                onPressed: () {
                  final levels = [StockLevel.inStock, StockLevel.lowStock, StockLevel.outOfStock];
                  final current = _stockLevels['product_103'] ?? StockLevel.inStock;
                  final next = levels[(levels.indexOf(current) + 1) % levels.length];
                  _simulateStockChange('product_103', 'MacBook Pro', next);
                },
              ),
              ActionChip(
                backgroundColor: const Color(0xFFF59E0B).withValues(alpha: 0.1),
                side: BorderSide(color: const Color(0xFFF59E0B).withValues(alpha: 0.3)),
                label: const Text(
                  'Trigger Deal Push',
                  style: TextStyle(color: Color(0xFFB45309), fontWeight: FontWeight.w600),
                ),
                onPressed: _triggerPromoDeal,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildProductCard(
    ThemeData theme, {
    required String id,
    required String name,
    required String retailer,
    required String image,
    required String price,
  }) {
    final level = _stockLevels[id] ?? StockLevel.inStock;

    return Card(
      elevation: 0,
      color: Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Colors.grey.shade200),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              width: 60,
              height: 60,
              decoration: BoxDecoration(
                color: Colors.grey.shade50,
                borderRadius: BorderRadius.circular(12),
              ),
              alignment: Alignment.center,
              child: Text(image, style: const TextStyle(fontSize: 32)),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    retailer,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: Colors.grey.shade600,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    price,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: const Color(0xFF1A56DB),
                    ),
                  ),
                ],
              ),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                StockBadge(
                  stockLevel: level,
                  showUpdatedText: true,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class RetailerDetailStubScreen extends StatelessWidget {
  final String retailerId;

  const RetailerDetailStubScreen({super.key, required this.retailerId});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Store Detail: $retailerId'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.store_rounded, size: 64, color: Color(0xFF1A56DB)),
            const SizedBox(height: 16),
            Text(
              'Detail Screen for $retailerId',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'This is a placeholder for the fully loaded retailer detail screen.',
              style: TextStyle(color: Colors.black54),
            ),
          ],
        ),
      ),
    );
  }
}
