import torch 
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    batch_data = {
        'scan0_ts': [],
        'scan1_ts': [],
        'scan0': [],
        'scan1': [],
        'imu_ts': [],
        'imu_dts': [],
        'accels': [],
        'gyros': [],
        'valid_length': [],
        'gt_pose0': [],
        'gt_pose1': [],
        'gt_velocity': [],
    }
    for sample in batch:
        batch_data['scan0_ts'].append(torch.tensor(sample['scan0_ts'], dtype=torch.float64))
        batch_data['scan1_ts'].append(torch.tensor(sample['scan1_ts'], dtype=torch.float64))
        batch_data['scan0'].append(torch.tensor(sample['scan0'], dtype=torch.float32))
        batch_data['scan1'].append(torch.tensor(sample['scan1'], dtype=torch.float32))
        batch_data['imu_ts'].append(torch.tensor(sample['imu_ts'], dtype=torch.float64))
        batch_data['imu_dts'].append(torch.tensor(sample['imu_dts'], dtype=torch.float32))
        batch_data['accels'].append(torch.tensor(sample['accels'], dtype=torch.float32))
        batch_data['gyros'].append(torch.tensor(sample['gyros'], dtype=torch.float32))
        batch_data['valid_length'].append(torch.tensor(sample['valid_length'], dtype=torch.int32))
        batch_data['gt_pose0'].append(torch.tensor(sample['gt_pose0'], dtype=torch.float32))
        batch_data['gt_pose1'].append(torch.tensor(sample['gt_pose1'], dtype=torch.float32))
        batch_data['gt_velocity'].append(torch.tensor(sample['gt_velocity'], dtype=torch.float32))

    batch_data['scan0'] = pad_sequence(batch_data['scan0'], batch_first=True, padding_value=0)
    batch_data['scan1'] = pad_sequence(batch_data['scan1'], batch_first=True, padding_value=0)
    batch_data['scan0_ts'] = torch.stack(batch_data['scan0_ts'])
    batch_data['scan1_ts'] = torch.stack(batch_data['scan1_ts'])
    batch_data['imu_ts'] = torch.stack(batch_data['imu_ts'])
    batch_data['imu_dts'] = torch.stack(batch_data['imu_dts'])
    batch_data['accels'] = torch.stack(batch_data['accels'])
    batch_data['gyros'] = torch.stack(batch_data['gyros'])
    batch_data['valid_length'] = torch.stack(batch_data['valid_length'])
    batch_data['gt_pose0'] = torch.stack(batch_data['gt_pose0'])
    batch_data['gt_pose1'] = torch.stack(batch_data['gt_pose1'])
    return batch_data


def collate_fn_with_txt(batch):
    batch_data = {
        'scan0_ts': [],
        'scan1_ts': [],
        'scan0': [],
        'scan1': [],
        'imu_ts': [],
        'imu_dts': [],
        'accels': [],
        'gyros': [],
        'valid_length': [],
        'gt_pose0': [],
        'gt_pose1': [],
        'icp_global_pose0': [],
        'icp_global_pose1': [],
        'icp_global_ts0': [],
        'icp_global_ts1': [],
        'icp_relative_pose': [],
    }
    for sample in batch:
        batch_data['scan0_ts'].append(torch.tensor(sample['scan0_ts'], dtype=torch.float64))
        batch_data['scan1_ts'].append(torch.tensor(sample['scan1_ts'], dtype=torch.float64))
        batch_data['scan0'].append(torch.tensor(sample['scan0'], dtype=torch.float32))
        batch_data['scan1'].append(torch.tensor(sample['scan1'], dtype=torch.float32))
        batch_data['imu_ts'].append(torch.tensor(sample['imu_ts'], dtype=torch.float64))
        batch_data['imu_dts'].append(torch.tensor(sample['imu_dts'], dtype=torch.float32))
        batch_data['accels'].append(torch.tensor(sample['accels'], dtype=torch.float32))
        batch_data['gyros'].append(torch.tensor(sample['gyros'], dtype=torch.float32))
        batch_data['valid_length'].append(torch.tensor(sample['valid_length'], dtype=torch.int32))
        batch_data['gt_pose0'].append(torch.tensor(sample['gt_pose0'], dtype=torch.float32))
        batch_data['gt_pose1'].append(torch.tensor(sample['gt_pose1'], dtype=torch.float32))
        batch_data['icp_global_pose0'].append(torch.tensor(sample['icp_global_pose0'], dtype=torch.float32))
        batch_data['icp_global_pose1'].append(torch.tensor(sample['icp_global_pose1'], dtype=torch.float32))
        batch_data['icp_global_ts0'].append(torch.tensor(sample['icp_global_ts0'], dtype=torch.float64))
        batch_data['icp_global_ts1'].append(torch.tensor(sample['icp_global_ts1'], dtype=torch.float64))
        batch_data['icp_relative_pose'].append(torch.tensor(sample['icp_relative_pose'], dtype=torch.float32))

    batch_data['scan0'] = pad_sequence(batch_data['scan0'], batch_first=True, padding_value=0)
    batch_data['scan1'] = pad_sequence(batch_data['scan1'], batch_first=True, padding_value=0)
    batch_data['scan0_ts'] = torch.stack(batch_data['scan0_ts'])
    batch_data['scan1_ts'] = torch.stack(batch_data['scan1_ts'])
    batch_data['imu_ts'] = torch.stack(batch_data['imu_ts'])
    batch_data['imu_dts'] = torch.stack(batch_data['imu_dts'])
    batch_data['accels'] = torch.stack(batch_data['accels'])
    batch_data['gyros'] = torch.stack(batch_data['gyros'])
    batch_data['valid_length'] = torch.stack(batch_data['valid_length'])
    batch_data['gt_pose0'] = torch.stack(batch_data['gt_pose0'])
    batch_data['gt_pose1'] = torch.stack(batch_data['gt_pose1'])
    batch_data['icp_global_pose0'] = torch.stack(batch_data['icp_global_pose0'])
    batch_data['icp_global_pose1'] = torch.stack(batch_data['icp_global_pose1'])
    batch_data['icp_global_ts0'] = torch.stack(batch_data['icp_global_ts0'])
    batch_data['icp_global_ts1'] = torch.stack(batch_data['icp_global_ts1'])
    batch_data['icp_relative_pose'] = torch.stack(batch_data['icp_relative_pose'])
    return batch_data
