import 'package:freezed_annotation/freezed_annotation.dart';

part 'stock_update_message.freezed.dart';
part 'stock_update_message.g.dart';

enum StockLevel { inStock, lowStock, outOfStock }

@freezed
class StockUpdateMessage with _$StockUpdateMessage {
  const factory StockUpdateMessage({
    required String productId,
    required String productName,
    required String retailerId,
    required StockLevel stockLevel,
    required DateTime updatedAt,
  }) = _StockUpdateMessage;

  factory StockUpdateMessage.fromJson(Map<String, dynamic> json) =>
      _$StockUpdateMessageFromJson(json);
}
