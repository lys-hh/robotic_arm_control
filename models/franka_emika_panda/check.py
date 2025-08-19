#!/usr/bin/env python3
"""
Franka Emika Panda URDF和MJCF XML对齐检查工具
用于验证frankaEmikaPanda.urdf文件与panda.xml文件是否一一对应
"""

import os
import xml.etree.ElementTree as ET
import numpy as np

dir_path = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(dir_path, 'frankaEmikaPanda.urdf')
XML_PATH = os.path.join(dir_path, 'panda.xml')

def extract_urdf_joints(urdf_file):
    """提取URDF关节信息"""
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    joints = []
    for joint in root.findall('.//joint'):
        if joint.get('type') == 'revolute':
            name = joint.get('name')
            
            # 提取origin
            origin = joint.find('origin')
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin is not None:
                xyz_str = origin.get('xyz', '0 0 0')
                rpy_str = origin.get('rpy', '0 0 0')
                xyz = [float(x) for x in xyz_str.split()]
                rpy = [float(x) for x in rpy_str.split()]
            
            # 提取axis
            axis = joint.find('axis')
            axis_xyz = [0, 0, 1]
            if axis is not None:
                axis_str = axis.get('xyz', '0 0 1')
                axis_xyz = [float(x) for x in axis_str.split()]
            
            # 提取limit
            limit = joint.find('limit')
            lower = -6.283185307179586
            upper = 6.283185307179586
            if limit is not None:
                lower = float(limit.get('lower', '-6.283185307179586'))
                upper = float(limit.get('upper', '6.283185307179586'))
            
            joints.append({
                'name': name,
                'xyz': xyz,
                'rpy': rpy,
                'axis': axis_xyz,
                'lower': lower,
                'upper': upper
            })
    
    return joints

def extract_xml_joints(xml_file):
    """提取XML关节信息"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    joints = []
    for joint in root.findall('.//joint'):
        name = joint.get('name')
        if name and name.startswith('joint'):  # 只处理主要关节，不包括手指关节
            axis_str = joint.get('axis', '0 0 1')
            axis = [float(x) for x in axis_str.split()]
            
            # 从range属性获取限位信息
            range_str = joint.get('range', '')
            lower = -2.8973  # 默认值
            upper = 2.8973   # 默认值
            
            if range_str:
                range_values = [float(x) for x in range_str.split()]
                if len(range_values) == 2:
                    lower, upper = range_values
            
            joints.append({
                'name': name,
                'axis': axis,
                'lower': lower,
                'upper': upper
            })
    
    return joints

def extract_xml_bodies(xml_file):
    """提取XML body位置信息"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    bodies = {}
    for body in root.findall('.//body'):
        name = body.get('name')
        if name:
            pos_str = body.get('pos', '0 0 0')
            quat_str = body.get('quat', '1 0 0 0')
            
            pos = [float(x) for x in pos_str.split()]
            quat = [float(x) for x in quat_str.split()]
            
            bodies[name] = {
                'pos': pos,
                'quat': quat
            }
    
    return bodies

def map_joint_names(urdf_joints, xml_joints):
    """映射关节名称"""
    # URDF关节名称到XML关节名称的映射
    name_mapping = {
        'panda_joint1': 'joint1',
        'panda_joint2': 'joint2', 
        'panda_joint3': 'joint3',
        'panda_joint4': 'joint4',
        'panda_joint5': 'joint5',
        'panda_joint6': 'joint6',
        'panda_joint7': 'joint7'
    }
    
    mapped_urdf_joints = []
    mapped_xml_joints = []
    
    for urdf_joint in urdf_joints:
        if urdf_joint['name'] in name_mapping:
            xml_name = name_mapping[urdf_joint['name']]
            # 找到对应的XML关节
            for xml_joint in xml_joints:
                if xml_joint['name'] == xml_name:
                    mapped_urdf_joints.append(urdf_joint)
                    mapped_xml_joints.append(xml_joint)
                    break
    
    return mapped_urdf_joints, mapped_xml_joints

def main():
    """主函数"""
    print("Franka Emika Panda URDF和MJCF XML对齐检查")
    print("=" * 60)
    
    # 提取信息
    print("正在解析URDF文件...")
    urdf_joints = extract_urdf_joints(URDF_PATH)
    
    print("正在解析XML文件...")
    xml_joints = extract_xml_joints(XML_PATH)
    xml_bodies = extract_xml_bodies(XML_PATH)
    
    # 映射关节名称
    mapped_urdf_joints, mapped_xml_joints = map_joint_names(urdf_joints, xml_joints)
    
    # 对比关节信息
    print("\n关节信息对比:")
    print("-" * 40)
    
    print(f"URDF关节数量: {len(urdf_joints)}")
    print(f"XML关节数量: {len(xml_joints)}")
    print(f"映射后关节数量: {len(mapped_urdf_joints)}")
    
    urdf_names = [j['name'] for j in mapped_urdf_joints]
    xml_names = [j['name'] for j in mapped_xml_joints]
    
    print(f"\n关节名称映射:")
    for i, (urdf_name, xml_name) in enumerate(zip(urdf_names, xml_names)):
        print(f"  {urdf_name} -> {xml_name}")
    
    # 对比关节限位
    print("\n关节限位对比:")
    print("-" * 40)
    
    for i, (urdf_joint, xml_joint) in enumerate(zip(mapped_urdf_joints, mapped_xml_joints)):
        print(f"\n关节 {i+1}: {urdf_joint['name']} -> {xml_joint['name']}")
        print(f"  URDF: lower={urdf_joint['lower']:.4f}, upper={urdf_joint['upper']:.4f}")
        print(f"  XML:  lower={xml_joint['lower']:.4f}, upper={xml_joint['upper']:.4f}")
        
        lower_diff = abs(urdf_joint['lower'] - xml_joint['lower'])
        upper_diff = abs(urdf_joint['upper'] - xml_joint['upper'])
        
        if lower_diff > 0.001 or upper_diff > 0.001:
            print(f"  ⚠️  限位不匹配! 差异: lower={lower_diff:.6f}, upper={upper_diff:.6f}")
        else:
            print(f"  ✅ 限位匹配")
    
    # 对比关节轴方向
    print("\n关节轴方向对比:")
    print("-" * 40)
    
    for i, (urdf_joint, xml_joint) in enumerate(zip(mapped_urdf_joints, mapped_xml_joints)):
        print(f"\n关节 {i+1}: {urdf_joint['name']} -> {xml_joint['name']}")
        print(f"  URDF: axis={urdf_joint['axis']}")
        print(f"  XML:  axis={xml_joint['axis']}")
        
        axis_diff = np.linalg.norm(np.array(urdf_joint['axis']) - np.array(xml_joint['axis']))
        if axis_diff > 0.001:
            print(f"  ⚠️  轴方向不匹配! 差异={axis_diff:.6f}")
        else:
            print(f"  ✅ 轴方向匹配")
    
    # 对比关键body位置
    print("\n关键body位置对比:")
    print("-" * 40)
    
    # Franka Panda的关键link名称
    key_bodies = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 
                  'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7']
    
    for i, body_name in enumerate(key_bodies):
        if body_name in xml_bodies:
            print(f"\n{body_name}:")
            print(f"  XML pos: {xml_bodies[body_name]['pos']}")
            print(f"  XML quat: {xml_bodies[body_name]['quat']}")
            
            # 对应的URDF关节位置
            if i < len(mapped_urdf_joints):
                print(f"  URDF xyz: {mapped_urdf_joints[i]['xyz']}")
                print(f"  URDF rpy: {mapped_urdf_joints[i]['rpy']}")
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结:")
    print("-" * 40)
    
    total_joints = len(mapped_urdf_joints)
    matched_limits = 0
    matched_axes = 0
    
    for urdf_joint, xml_joint in zip(mapped_urdf_joints, mapped_xml_joints):
        # 检查限位匹配
        lower_diff = abs(urdf_joint['lower'] - xml_joint['lower'])
        upper_diff = abs(urdf_joint['upper'] - xml_joint['upper'])
        if lower_diff <= 0.001 and upper_diff <= 0.001:
            matched_limits += 1
        
        # 检查轴方向匹配
        axis_diff = np.linalg.norm(np.array(urdf_joint['axis']) - np.array(xml_joint['axis']))
        if axis_diff <= 0.001:
            matched_axes += 1
    
    print(f"总关节数: {total_joints}")
    print(f"限位匹配: {matched_limits}/{total_joints}")
    print(f"轴方向匹配: {matched_axes}/{total_joints}")
    
    if matched_limits == total_joints and matched_axes == total_joints:
        print("✅ 所有关节参数完全匹配!")
    else:
        print("⚠️  存在不匹配的关节参数")
    
    print("\n" + "=" * 60)
    print("检查完成!")
    print("=" * 60)

if __name__ == "__main__":
    main() 