DataRoot = {}
DataRoot['local'] = {
                        'sceneflow':    '/home/amigo/tmp/data/sceneflow',
                        'tartan':       '/home/amigo/tmp/data/tartan',
                        'chairs':       '/home/amigo/tmp/data/flyingchairs',
                        'flying':       '/home/amigo/tmp/data/sceneflow',
                        'sintel':       '/home/amigo/tmp/data/sintel/training',
                        'euroc':        '/prague/tartanvo_data/euroc',
                        'kitti-stereo': '/prague/tartanvo_data/kitti/stereo', 
                        'kitti-vo':     '/prague/tartanvo_data/kitti/vo'
}


DataRoot['cluster'] = {
                        'sceneflow':    '/data/datasets/yaoyuh/StereoData/SceneFlow',
                        'tartan':       '/data/datasets/wenshanw/tartan_data',
                        'tartan-c':     '/project/learningvo/tartanair_v1_5',
                        'chairs':       '/project/learningvo/flowdata/FlyingChairs_release',
                        'flying':       '/data/datasets/yaoyuh/StereoData/SceneFlow',
                        'sintel':       '/project/learningvo/flowdata/sintel/training',
                        'euroc':        '/project/learningvo/euroc',
                        'kitti-stereo': '/project/learningvo/stereo_data/kitti/training', 
                        'tartan-z':     '/scratch/learningvo/tartanair_v1_5',
                        'chairs-z':     '/scratch/learningvo/flyingchairs',
                        'sintel-z':     '/scratch/learningvo/sintel/training',
                        'flying-z':     '/scratch/learningvo/SceneFlow',
}

DataRoot['dgx'] = {
                        'sceneflow':    '/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow',
                        'tartan-c':     '/tmp2/wenshan/tartanair_v1_5',
                        'chairs':       '/tmp2/wenshan/flyingchairs',
                        'flying':       '/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow',
                        'sintel':       '/tmp2/wenshan/sintel/training',
                        'kitti-stereo': '/tmp2/wenshan/kitti/training', 
}

DataRoot['psc'] = {
    
}




STEREO_DR = {'sceneflow':   {'local':   ['/home/amigo/tmp/data/sceneflow', '/home/amigo/tmp/data/sceneflow'],
                            'cluster':  ['/data/datasets/yaoyuh/StereoData/SceneFlow', '/data/datasets/yaoyuh/StereoData/SceneFlow'],
                            'azure':    ['SceneFlow', 'SceneFlow'],
                            'dgx':      ['/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow', '/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/SceneFlow','/ocean/projects/cis210086p/wenshanw/SceneFlow'],
                            }, 
            'tartan':       {'local':   ['/home/amigo/tmp/data/tartan', '/home/amigo/tmp/data/tartan'],
                            'local_test':  ['/peru/tartanair', '/peru/tartanair'],
                            'cluster':  ['/data/datasets/wenshanw/tartan_data', '/data/datasets/wenshanw/tartan_data'],
                            'cluster2':  ['/project/learningvo/tartanair_v1_5', '/project/learningvo/tartanair_v1_5'],
                            'azure':    ['', ''],
                            'dgx':      ['/tmp2/wenshan/tartanair_v1_5', '/tmp2/wenshan/tartanair_v1_5'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/tartanair_v1_5','/ocean/projects/cis210086p/wenshanw/tartanair_v1_5'],
                            },
            'kitti':       {'local':    ['/prague/tartanvo_data/kitti/stereo', '/prague/tartanvo_data/kitti/stereo'], # DEBIG: stereo
                            'cluster':  ['/project/learningvo/stereo_data/kitti/training', '/project/learningvo/stereo_data/kitti/training'],
                            'azure':    ['', ''], # NO KITTI on AZURE yet!!
                            'dgx':      ['/tmp2/wenshan/kitti/training', '/tmp2/wenshan/kitti/training'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/kitti/training','/ocean/projects/cis210086p/wenshanw/kitti/training'],
                            },
            'euroc':       {'local':   ['/prague/tartanvo_data/euroc', '/prague/tartanvo_data/euroc'],
                            },
            }


# Datasets for FlowVo
FLOWVO_DR = {'tartan':      {'local':   '/home/amigo/tmp/data/tartan', # '/home/amigo/tmp/data/tartanair_pose_and_imu',# 
                            'local2':   '/home/amigo/tmp/data/tartanair_pose_and_imu', #'/cairo/tartanair_test_cvpr', # '/home/amigo/tmp/data/tartan', # 
                            'local_test':  '/peru/tartanair',
                            'cluster':  '/data/datasets/wenshanw/tartan_data',
                            'cluster2':  '/project/learningvo/tartanair_v1_5',
                            'azure':    '',
                            'dgx':      '/tmp2/wenshan/tartanair_v1_5',
                            'psc':      '/ocean/projects/cis210086p/wenshanw/tartanair_v1_5',
                            }, 
             'euroc':       {'local':   '/prague/tartanvo_data/euroc', 
                            'cluster2':  '/project/learningvo/euroc',
                            },
             'kitti':       {'local':   '/prague/tartanvo_data/kitti/vo', 
                            },
}

# Datasets for Flow
FLOW_DR =   {'flyingchairs':{'local':   ['/home/amigo/tmp/data/flyingchairs', '/home/amigo/tmp/data/flyingchairs'],
                            'cluster':  ['/project/learningvo/flowdata/FlyingChairs_release', '/project/learningvo/flowdata/FlyingChairs_release'],
                            'azure':    ['FlyingChairs_release', 'FlyingChairs_release'],
                            'dgx':      ['/tmp2/wenshan/flyingchairs', '/tmp2/wenshan/flyingchairs'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/flyingchairs','/ocean/projects/cis210086p/wenshanw/flyingchairs'],
                            }, 
            'flyingthings': {'local':   ['/home/amigo/tmp/data/sceneflow', '/home/amigo/tmp/data/sceneflow/frames_cleanpass'],
                            'cluster':  ['/data/datasets/yaoyuh/StereoData/SceneFlow', '/project/learningvo/flowdata/optical_flow'],
                            'azure':    ['SceneFlow','SceneFlow'],
                            'dgx':      ['/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow', '/tmp2/wenshan/optical_flow'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/SceneFlow','/ocean/projects/cis210086p/wenshanw/optical_flow'],
                            }, 
            'sintel':       {'local':   ['/home/amigo/tmp/data/sintel/training', '/home/amigo/tmp/data/sintel/training'],
                            'cluster':  ['/project/learningvo/flowdata/sintel/training', '/project/learningvo/flowdata/sintel/training'],
                            'azure':    ['sintel/training', 'sintel/training'],
                            'dgx':      ['/tmp2/wenshan/sintel/training', '/tmp2/wenshan/sintel/training'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/sintel/training','/ocean/projects/cis210086p/wenshanw/sintel/training'],
                            }, 
            'tartan':       {'local':   ['/home/amigo/tmp/data/tartan', '/home/amigo/tmp/data/tartan'],
                            'local_test':  ['/peru/tartanair', '/peru/tartanair'],
                            'cluster':  ['/data/datasets/wenshanw/tartan_data', '/data/datasets/wenshanw/tartan_data'],
                            'cluster2':  ['/project/learningvo/tartanair_v1_5', '/project/learningvo/tartanair_v1_5'],
                            'azure':    ['', ''],
                            'dgx':      ['/tmp2/wenshan/tartanair_v1_5', '/tmp2/wenshan/tartanair_v1_5'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/tartanair_v1_5','/ocean/projects/cis210086p/wenshanw/tartanair_v1_5'],
                            }, 
            'euroc':        {'local':   ['/prague/tartanvo_data/euroc', '/prague/tartanvo_data/euroc'],
                            'cluster2':  ['/project/learningvo/euroc', '/project/learningvo/euroc'],
                            },
            'kitti':        {'local':   ['/prague/tartanvo_data/kitti/vo', '/prague/tartanvo_data/kitti/vo'],
                            },
    
}

