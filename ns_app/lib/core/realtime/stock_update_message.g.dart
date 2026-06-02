// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'stock_update_message.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$StockUpdateMessageImpl _$$StockUpdateMessageImplFromJson(
  Map<String, dynamic> json,
) => _$StockUpdateMessageImpl(
  productId: json['productId'] as String,
  productName: json['productName'] as String,
  retailerId: json['retailerId'] as String,
  stockLevel: $enumDecode(_$StockLevelEnumMap, json['stockLevel']),
  updatedAt: DateTime.parse(json['updatedAt'] as String),
);

Map<String, dynamic> _$$StockUpdateMessageImplToJson(
  _$StockUpdateMessageImpl instance,
) => <String, dynamic>{
  'productId': instance.productId,
  'productName': instance.productName,
  'retailerId': instance.retailerId,
  'stockLevel': _$StockLevelEnumMap[instance.stockLevel]!,
  'updatedAt': instance.updatedAt.toIso8601String(),
};

const _$StockLevelEnumMap = {
  StockLevel.inStock: 'inStock',
  StockLevel.lowStock: 'lowStock',
  StockLevel.outOfStock: 'outOfStock',
};
