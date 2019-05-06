# Rendering Tensor Generator

Requirement:
 - Python2.7
 - Chumpy
 - tensorflow
 - Opendr
 - tqdm
 - opencv-python

Usage:
- Download the [Market-1501 dataset](http://www.liangzheng.com.cn/Project/project_reid.html)

- Download the pre-trained hmr network parameters to ```./hmr```:
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```

- Modify input directory (in Market1501 format) and output directory in ```generated_render_tensor.sh```

- Run the rendering tensor generating script:

```bash
bash generated_render_tensor.sh
```
It will take several hours before generating all of the rendering tensors.
