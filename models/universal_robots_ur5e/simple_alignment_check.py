#!/usr/bin/env python3
"""
简化的URDF和MJCF XML对齐检查工具
"""

import os
import xml.etree.ElementTree as ET
import numpy as np

dir_path = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(dir_path, 'ur5e.urdf')
XML_PATH = os.path.join(dir_path, 'ur5e.xml')

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
        axis_str = joint.get('axis', '0 0 1')
        axis = [float(x) for x in axis_str.split()]
        
        # 从class中获取range信息
        joint_class = joint.get('class', '')
        lower = -6.28319
        upper = 6.28319
        
        if 'size3_limited' in joint_class:
            lower = -3.1415
            upper = 3.1415
        
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

def main():
    """主函数"""
    print("URDF和MJCF XML对齐检查")
    print("=" * 50)
    
    # 提取信息
    print("正在解析URDF文件...")
    urdf_joints = extract_urdf_joints(URDF_PATH)
    
    print("正在解析XML文件...")
    xml_joints = extract_xml_joints(XML_PATH)
    xml_bodies = extract_xml_bodies(XML_PATH)
    
    # 对比关节信息
    print("\n关节信息对比:")
    print("-" * 30)
    
    print(f"URDF关节数量: {len(urdf_joints)}")
    print(f"XML关节数量: {len(xml_joints)}")
    
    urdf_names = [j['name'] for j in urdf_joints]
    xml_names = [j['name'] for j in xml_joints]
    
    print(f"\n关节名称:")
    print(f"URDF: {urdf_names}")
    print(f"XML:  {xml_names}")
    print(f"名称匹配: {urdf_names == xml_names}")
    
    # 对比关节限位
    print("\n关节限位对比:")
    print("-" * 30)
    
    for i, (urdf_joint, xml_joint) in enumerate(zip(urdf_joints, xml_joints)):
        print(f"\n关节 {i+1}: {urdf_joint['name']}")
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
    print("-" * 30)
    
    for i, (urdf_joint, xml_joint) in enumerate(zip(urdf_joints, xml_joints)):
        print(f"\n关节 {i+1}: {urdf_joint['name']}")
        print(f"  URDF: axis={urdf_joint['axis']}")
        print(f"  XML:  axis={xml_joint['axis']}")
        
        axis_diff = np.linalg.norm(np.array(urdf_joint['axis']) - np.array(xml_joint['axis']))
        if axis_diff > 0.001:
            print(f"  ⚠️  轴方向不匹配! 差异={axis_diff:.6f}")
        else:
            print(f"  ✅ 轴方向匹配")
    
    # 对比关键body位置
    print("\n关键body位置对比:")
    print("-" * 30)
    
    key_bodies = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
    
    for i, body_name in enumerate(key_bodies):
        if body_name in xml_bodies:
            print(f"\n{body_name}:")
            print(f"  XML pos: {xml_bodies[body_name]['pos']}")
            print(f"  XML quat: {xml_bodies[body_name]['quat']}")
            
            # 对应的URDF关节位置
            if i < len(urdf_joints):
                print(f"  URDF xyz: {urdf_joints[i]['xyz']}")
                print(f"  URDF rpy: {urdf_joints[i]['rpy']}")
    
    print("\n" + "=" * 50)
    print("检查完成!")
    print("=" * 50)

if __name__ == "__main__":
    main() 