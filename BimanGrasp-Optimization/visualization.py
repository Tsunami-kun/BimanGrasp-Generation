import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go
import plotly.io as pio
from utils.hand_model import HandModel
from utils.object_model import ObjectModel

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='Curver_Storage_Bin_Black_Small')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--no_st', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--result_path', type=str, default='../data/experiments/test/results')

    args = parser.parse_args()

    device = 'cpu'

    left_hand_model = HandModel(
        mjcf_path='mjcf/left_shadow_hand.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/left_hand_contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        device=device,
        handedness = 'left_hand'
    )

    right_hand_model = HandModel(
        mjcf_path='mjcf/right_shadow_hand.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/right_hand_contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        device=device,
        handedness = 'right_hand'
    )
    
    object_model = ObjectModel(
        data_root_path='../data/meshdata',
        batch_size_each=1,
        num_samples=2000, 
        device=device,
        size='large'
    )
    # load results
    data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
    
    right_qpos = data_dict['qpos_right']
    right_rot = np.array(transforms3d.euler.euler2mat(*[right_qpos[name] for name in rot_names]))
    right_rot = right_rot[:, :2].T.ravel().tolist()
    right_hand_pose = torch.tensor([right_qpos[name] for name in translation_names] + right_rot + [right_qpos[name] for name in joint_names], dtype=torch.float, device=device)
    if 'qpos_right_st' in data_dict:
        right_qpos_st = data_dict['qpos_right_st']
        right_rot = np.array(transforms3d.euler.euler2mat(*[right_qpos_st[name] for name in rot_names]))
        right_rot = right_rot[:, :2].T.ravel().tolist()
        right_hand_pose_st = torch.tensor([right_qpos_st[name] for name in translation_names] + right_rot + [right_qpos_st[name] for name in joint_names], dtype=torch.float, device=device)

    # load left results
    left_qpos = data_dict['qpos_left']
    left_rot = np.array(transforms3d.euler.euler2mat(*[left_qpos[name] for name in rot_names]))
    left_rot = left_rot[:, :2].T.ravel().tolist()
    left_hand_pose = torch.tensor([left_qpos[name] for name in translation_names] + left_rot + [left_qpos[name] for name in joint_names], dtype=torch.float, device=device)
    if 'qpos_left_st' in data_dict:
        left_qpos_st = data_dict['qpos_left_st']
        left_rot = np.array(transforms3d.euler.euler2mat(*[left_qpos_st[name] for name in rot_names]))
        left_rot = left_rot[:, :2].T.ravel().tolist()
        left_hand_pose_st = torch.tensor([left_qpos_st[name] for name in translation_names] + left_rot + [left_qpos_st[name] for name in joint_names], dtype=torch.float, device=device)


    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    right_hand_model.set_parameters(right_hand_pose.unsqueeze(0))
    right_hand_en_plotly = right_hand_model.get_plotly_data(i=0, opacity=1, color='lightslategray', with_contact_points=False)
    
    left_hand_model.set_parameters(left_hand_pose.unsqueeze(0))
    left_hand_en_plotly = left_hand_model.get_plotly_data(i=0, opacity=1, color='lightslategray', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='seashell', opacity=1)    
       
    
    fig = go.Figure(right_hand_en_plotly + object_plotly + left_hand_en_plotly)

    fig.update_layout(
        paper_bgcolor='#E2F0D9',  
        plot_bgcolor='#E2F0D9'    
    )

    fig.update_layout(scene_aspectmode='data')    
    fig.update_layout(
    scene=dict(
        xaxis=dict(
            visible=False,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False
        ),
        zaxis=dict(
            visible=False,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False
        )
    )
    )
    
    fig.show()
