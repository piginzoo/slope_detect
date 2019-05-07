#倾斜检测

This is a pretty simple project. It will detect whether the bill picture is inclined.

It use VGG as backbone, then send the tensor to a very simple full connection layer, which is 256 dimensions, the last layer is a 4 classes output, represents 4 directions: no lean,  or lean 90' clockwise, or 180' or 270'.

Train.py is in charge of training work, you can run it by bin/train.sh, you can adjust the parameters in the script for those hyper-parameters.

Pred.py can help you predict the picture direction, it will load the model, restore all weights of the network, create a session then predict the picture.

There is also a web server, who did almost same function with pred.py, but more is, it will expose the restful api to other application. You can use it to combine with your application to do more things.

And in data_generators, you can find the code who help to create train fake data and labels, I did not upload the raw data, but if you need, i can share some. Just free to contact me by email: piginzoo@gmail.com.

piginzoo 2019.5
