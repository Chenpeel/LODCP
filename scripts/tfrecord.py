import tensorflow as tf

def create_tfrecord(images, masks, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    for img, mask in zip(images, masks):
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
            'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tobytes()]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
