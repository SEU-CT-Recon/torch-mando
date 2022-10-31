pip uninstall torch_mando -y 
rm -rf __pycache__
rm -rf objs
rm -rf build
rm -rf torch_mando.egg-info 
python setup.py install