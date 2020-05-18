
import tensorflow as tf

def test1():
    # data = tf.constant([[0.2, 0.2, 0.4, 0.2],[0.3, 0.5, 0.1, 0.1]])
    data = tf.constant([[0.2, 0.2, 0.4, 0.2]])
    print(tf.shape(data))
    a = tf.argmax(data,axis=1)
    with tf.Session() as s:
        b = s.run(a)
        print(b.shape)
        print(b)


def test2():
    image_label_list = [1,2,2,2,3,1,5,6,4,3,2]
    batch_num = 3
    #while True:
    for i in range(0, len(image_label_list), batch_num):
        batch = image_label_list[i:i + batch_num]
        #print(batch)
        if len(batch) >= batch_num:
            val = batch
        else:
            continue
        print(val)
    return val



if __name__ == '__main__':
    test1()
    #test2()
    # val = next(val)
    # print(val)




