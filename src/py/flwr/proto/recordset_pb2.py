# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: flwr/proto/recordset.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1a\x66lwr/proto/recordset.proto\x12\nflwr.proto\"\x1a\n\nDoubleList\x12\x0c\n\x04vals\x18\x01 \x03(\x01\"\x18\n\x08SintList\x12\x0c\n\x04vals\x18\x01 \x03(\x12\"\x18\n\x08UintList\x12\x0c\n\x04vals\x18\x01 \x03(\x04\"\x18\n\x08\x42oolList\x12\x0c\n\x04vals\x18\x01 \x03(\x08\"\x1a\n\nStringList\x12\x0c\n\x04vals\x18\x01 \x03(\t\"\x19\n\tBytesList\x12\x0c\n\x04vals\x18\x01 \x03(\x0c\"B\n\x05\x41rray\x12\r\n\x05\x64type\x18\x01 \x01(\t\x12\r\n\x05shape\x18\x02 \x03(\x05\x12\r\n\x05stype\x18\x03 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\x0c\"\xd8\x01\n\x12MetricsRecordValue\x12\x10\n\x06\x64ouble\x18\x01 \x01(\x01H\x00\x12\x10\n\x06sint64\x18\x02 \x01(\x12H\x00\x12\x10\n\x06uint64\x18\x03 \x01(\x04H\x00\x12-\n\x0b\x64ouble_list\x18\x15 \x01(\x0b\x32\x16.flwr.proto.DoubleListH\x00\x12)\n\tsint_list\x18\x16 \x01(\x0b\x32\x14.flwr.proto.SintListH\x00\x12)\n\tuint_list\x18\x17 \x01(\x0b\x32\x14.flwr.proto.UintListH\x00\x42\x07\n\x05value\"\x92\x03\n\x12\x43onfigsRecordValue\x12\x10\n\x06\x64ouble\x18\x01 \x01(\x01H\x00\x12\x10\n\x06sint64\x18\x02 \x01(\x12H\x00\x12\x10\n\x06uint64\x18\x03 \x01(\x04H\x00\x12\x0e\n\x04\x62ool\x18\x04 \x01(\x08H\x00\x12\x10\n\x06string\x18\x05 \x01(\tH\x00\x12\x0f\n\x05\x62ytes\x18\x06 \x01(\x0cH\x00\x12-\n\x0b\x64ouble_list\x18\x15 \x01(\x0b\x32\x16.flwr.proto.DoubleListH\x00\x12)\n\tsint_list\x18\x16 \x01(\x0b\x32\x14.flwr.proto.SintListH\x00\x12)\n\tuint_list\x18\x17 \x01(\x0b\x32\x14.flwr.proto.UintListH\x00\x12)\n\tbool_list\x18\x18 \x01(\x0b\x32\x14.flwr.proto.BoolListH\x00\x12-\n\x0bstring_list\x18\x19 \x01(\x0b\x32\x16.flwr.proto.StringListH\x00\x12+\n\nbytes_list\x18\x1a \x01(\x0b\x32\x15.flwr.proto.BytesListH\x00\x42\x07\n\x05value\"M\n\x10ParametersRecord\x12\x11\n\tdata_keys\x18\x01 \x03(\t\x12&\n\x0b\x64\x61ta_values\x18\x02 \x03(\x0b\x32\x11.flwr.proto.Array\"\x8f\x01\n\rMetricsRecord\x12\x31\n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32#.flwr.proto.MetricsRecord.DataEntry\x1aK\n\tDataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12-\n\x05value\x18\x02 \x01(\x0b\x32\x1e.flwr.proto.MetricsRecordValue:\x02\x38\x01\"\x8f\x01\n\rConfigsRecord\x12\x31\n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32#.flwr.proto.ConfigsRecord.DataEntry\x1aK\n\tDataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12-\n\x05value\x18\x02 \x01(\x0b\x32\x1e.flwr.proto.ConfigsRecordValue:\x02\x38\x01\"\x97\x03\n\tRecordSet\x12\x39\n\nparameters\x18\x01 \x03(\x0b\x32%.flwr.proto.RecordSet.ParametersEntry\x12\x33\n\x07metrics\x18\x02 \x03(\x0b\x32\".flwr.proto.RecordSet.MetricsEntry\x12\x33\n\x07\x63onfigs\x18\x03 \x03(\x0b\x32\".flwr.proto.RecordSet.ConfigsEntry\x1aO\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12+\n\x05value\x18\x02 \x01(\x0b\x32\x1c.flwr.proto.ParametersRecord:\x02\x38\x01\x1aI\n\x0cMetricsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12(\n\x05value\x18\x02 \x01(\x0b\x32\x19.flwr.proto.MetricsRecord:\x02\x38\x01\x1aI\n\x0c\x43onfigsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12(\n\x05value\x18\x02 \x01(\x0b\x32\x19.flwr.proto.ConfigsRecord:\x02\x38\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'flwr.proto.recordset_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_METRICSRECORD_DATAENTRY']._options = None
  _globals['_METRICSRECORD_DATAENTRY']._serialized_options = b'8\001'
  _globals['_CONFIGSRECORD_DATAENTRY']._options = None
  _globals['_CONFIGSRECORD_DATAENTRY']._serialized_options = b'8\001'
  _globals['_RECORDSET_PARAMETERSENTRY']._options = None
  _globals['_RECORDSET_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_RECORDSET_METRICSENTRY']._options = None
  _globals['_RECORDSET_METRICSENTRY']._serialized_options = b'8\001'
  _globals['_RECORDSET_CONFIGSENTRY']._options = None
  _globals['_RECORDSET_CONFIGSENTRY']._serialized_options = b'8\001'
  _globals['_DOUBLELIST']._serialized_start=42
  _globals['_DOUBLELIST']._serialized_end=68
  _globals['_SINTLIST']._serialized_start=70
  _globals['_SINTLIST']._serialized_end=94
  _globals['_UINTLIST']._serialized_start=96
  _globals['_UINTLIST']._serialized_end=120
  _globals['_BOOLLIST']._serialized_start=122
  _globals['_BOOLLIST']._serialized_end=146
  _globals['_STRINGLIST']._serialized_start=148
  _globals['_STRINGLIST']._serialized_end=174
  _globals['_BYTESLIST']._serialized_start=176
  _globals['_BYTESLIST']._serialized_end=201
  _globals['_ARRAY']._serialized_start=203
  _globals['_ARRAY']._serialized_end=269
  _globals['_METRICSRECORDVALUE']._serialized_start=272
  _globals['_METRICSRECORDVALUE']._serialized_end=488
  _globals['_CONFIGSRECORDVALUE']._serialized_start=491
  _globals['_CONFIGSRECORDVALUE']._serialized_end=893
  _globals['_PARAMETERSRECORD']._serialized_start=895
  _globals['_PARAMETERSRECORD']._serialized_end=972
  _globals['_METRICSRECORD']._serialized_start=975
  _globals['_METRICSRECORD']._serialized_end=1118
  _globals['_METRICSRECORD_DATAENTRY']._serialized_start=1043
  _globals['_METRICSRECORD_DATAENTRY']._serialized_end=1118
  _globals['_CONFIGSRECORD']._serialized_start=1121
  _globals['_CONFIGSRECORD']._serialized_end=1264
  _globals['_CONFIGSRECORD_DATAENTRY']._serialized_start=1189
  _globals['_CONFIGSRECORD_DATAENTRY']._serialized_end=1264
  _globals['_RECORDSET']._serialized_start=1267
  _globals['_RECORDSET']._serialized_end=1674
  _globals['_RECORDSET_PARAMETERSENTRY']._serialized_start=1445
  _globals['_RECORDSET_PARAMETERSENTRY']._serialized_end=1524
  _globals['_RECORDSET_METRICSENTRY']._serialized_start=1526
  _globals['_RECORDSET_METRICSENTRY']._serialized_end=1599
  _globals['_RECORDSET_CONFIGSENTRY']._serialized_start=1601
  _globals['_RECORDSET_CONFIGSENTRY']._serialized_end=1674
# @@protoc_insertion_point(module_scope)
