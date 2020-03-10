def load_class_dict(label_file: str):
    """
    Reads the class label file and returns a tuple consisting of a list of classes, a list of colors
    and a dictionary {class: color}
    :param label_file: The text file containg the class names and colors
    :return: A tuple of classes, colors and a dicitonary {class: color}
    """
    with open(label_file, 'r') as tf:
        text = tf.readlines()
    classes_file = [t.strip() for t in text]

    classes = [l.split(';')[0] for l in classes_file]
    colors = [[int(l.split(';')[n]) for n in range(1, len(l.split(';')))] for l in classes_file]
    class_dict = {classes[n]: colors[n] for n in range(len(classes))}

    return classes, colors, class_dict