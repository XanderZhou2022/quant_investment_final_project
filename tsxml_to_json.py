#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天软 TSXML 转 JSON 转换器
"""

import xml.etree.ElementTree as ET
import json
import re
from datetime import datetime, timedelta

def excel_serial_to_date(serial):
    """
    将 Excel 序列日转换为 YYYY-MM-DD 格式
    Excel 1900 日期系统：1900-01-01 = 1
    注意：Excel 错误地认为 1900 年是闰年，所以 1900-03-01 实际上是序列日 61
    """
    try:
        serial = float(serial)
        # Excel 基准日期：1900-01-01（但 Excel 把 1900-02-29 当作有效日期）
        # Python 的 datetime 从 1900-01-01 开始，但需要减去 2 天来修正 Excel 的闰年 bug
        # 实际上，更准确的做法是：如果 serial >= 60，减去 1（因为 Excel 有 1900-02-29）
        base_date = datetime(1899, 12, 30)  # 这样 serial=1 对应 1900-01-01
        
        if serial >= 60:
            # Excel 错误地将 1900 年当作闰年，所以序列日 >= 60 需要减去 1
            serial = serial - 1
        
        target_date = base_date + timedelta(days=int(serial))
        return target_date.strftime('%Y-%m-%d')
    except (ValueError, OverflowError):
        return None

def is_date_field(field_name):
    """判断字段名是否为日期字段"""
    date_keywords = ['开始日', '截止日', '起始日', '终止日', '因子截止日', '回测开始日', '回测截止日']
    # 由于编码问题，这里使用包含检查
    return any(keyword in str(field_name) for keyword in date_keywords)

def parse_data_value(data_elem, warnings):
    """解析 DATA 元素的值"""
    data_type = data_elem.get('TYPE', '').upper()
    value = data_elem.get('VALUE', '')
    
    # 缺失值处理
    if value == '-':
        return None
    
    # 类型转换
    if data_type == 'STRING':
        return value
    elif data_type == 'INTEGER':
        try:
            return int(value)
        except (ValueError, TypeError):
            warnings.append(f"无法将 INTEGER 值 '{value}' 转换为整数，保留为字符串")
            return value
    elif data_type == 'DOUBLE':
        try:
            return float(value)
        except (ValueError, TypeError):
            warnings.append(f"无法将 DOUBLE 值 '{value}' 转换为浮点数，保留为字符串")
            return value
    elif data_type == 'ARRAY':
        return parse_array(data_elem, warnings)
    else:
        warnings.append(f"未知的 DATA TYPE: {data_type}，保留原值")
        return value

def parse_array(data_elem, warnings):
    """解析 ARRAY 类型的 DATA 元素"""
    items = data_elem.findall('ITEM')
    
    # 检查是否所有索引都是 INTEGER（顺序索引）
    all_integer_indices = True
    for item in items:
        index_elem = item.find('INDEX')
        if index_elem is None:
            warnings.append("数组 ITEM 缺少 INDEX 元素")
            all_integer_indices = False
            break
        index_type = index_elem.get('TYPE', '')
        if index_type != 'INTEGER':
            all_integer_indices = False
            break
    
    if all_integer_indices:
        # 转换为 JSON 数组
        result = []
        for item in items:
            data_elem = item.find('DATA')
            if data_elem is not None:
                value = parse_data_value(data_elem, warnings)
                result.append(value)
            else:
                warnings.append("数组 ITEM 缺少 DATA 元素")
                result.append(None)
        return result
    else:
        # 转换为 JSON 对象
        result = {}
        for item in items:
            index_elem = item.find('INDEX')
            if index_elem is None:
                warnings.append("对象 ITEM 缺少 INDEX 元素")
                continue
            
            index_value = index_elem.get('VALUE', '')
            data_elem = item.find('DATA')
            
            if data_elem is not None:
                value = parse_data_value(data_elem, warnings)
                
                # 检查是否是日期字段且值为 double
                if is_date_field(index_value) and isinstance(value, (int, float)):
                    # 保留原数值
                    result[index_value] = value
                    # 添加字符串日期字段
                    date_str = excel_serial_to_date(value)
                    if date_str:
                        result[index_value + '_str'] = date_str
                    else:
                        warnings.append(f"无法将序列日 {value} 转换为日期字符串（字段：{index_value}）")
                else:
                    result[index_value] = value
            else:
                warnings.append(f"对象 ITEM 缺少 DATA 元素（索引：{index_value}）")
                result[index_value] = None
        
        return result

def tsxml_to_json(xml_file_path):
    """将 TSXML 文件转换为 JSON"""
    warnings = []
    
    try:
        # 读取 XML 文件（GB2312 编码）
        with open(xml_file_path, 'r', encoding='gb2312', errors='ignore') as f:
            content = f.read()
        
        # 解析 XML
        root = ET.fromstring(content)
        
        # 检查根元素类型
        if root.tag != 'TSXML' or root.get('TYPE') != 'ARRAY':
            warnings.append("根元素不是 TSXML TYPE='ARRAY'")
            return None, warnings
        
        # 解析顶层数组
        result = parse_array(root, warnings)
        
        # 如果有警告，添加到结果中
        if warnings:
            if isinstance(result, dict):
                result['__warnings'] = warnings
            else:
                # 如果顶层是数组，转换为对象
                result = {
                    '__data': result,
                    '__warnings': warnings
                }
        
        return result, warnings
    
    except ET.ParseError as e:
        warnings.append(f"XML 解析错误: {str(e)}")
        return None, warnings
    except Exception as e:
        warnings.append(f"转换过程出错: {str(e)}")
        return None, warnings

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python tsxml_to_json.py <xml_file> [output_file]")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result, warnings = tsxml_to_json(xml_file)
    
    if result is None:
        print("转换失败，警告信息：")
        for w in warnings:
            print(f"  - {w}")
        sys.exit(1)
    
    # 输出 JSON
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"JSON 已保存到: {output_file}")
    else:
        print(json_str)
    
    if warnings:
        print("\n警告信息：")
        for w in warnings:
            print(f"  - {w}")
