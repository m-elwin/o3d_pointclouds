from setuptools import setup

package_name = 'o3d_pointcloud'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name,
         ['package.xml', 'launch/table_demo.launch.xml', 'config/pcl.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='elwin',
    maintainer_email='elwin@northwestern.edu',
    description='Demonstrate Point Cloud Processing with Open3D',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'table = o3d_pointcloud.tablefinder:table_entry'
        ],
    },
)
