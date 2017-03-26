from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import getcwd
import csv
from keras.layers.merge import Concatenate
from keras.engine import Input

def displayCV2(img):
    '''
    Display a CV2 Image
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_img_for_visualization(image, angle, pred_angle, frame):
    '''
    Used by visualize_dataset method to format image prior to displaying. Converts colorspace back to original BGR,
    applies text to display steering angle and frame number (within batch to be visualized),
    and applies lines representing steering angle and model-predicted steering angle (if available) to image.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[0:2]
    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2, 18), fontFace=font, fontScale=.5, color=(200, 100, 100),
                thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2, 33), fontFace=font, fontScale=.5, color=(200, 100, 100),
                thickness=1)
    # apply a line representing the steering angle
    cv2.line(img, (int(w / 2), int(h)), (int(w / 2 + angle * w / 4), int(h / 2)), (0, 255, 0), thickness=4)
    if pred_angle is not None:
        cv2.line(img, (int(w / 2), int(h)), (int(w / 2 + pred_angle * w / 4), int(h / 2)), (0, 0, 255), thickness=4)
    return img


def visualize_dataset(X, y, y_pred=None):
    '''
    format the data from the dataset (image, steering angle) and display
    '''
    for i in range(len(X)):
        if y_pred is not None:
            img = process_img_for_visualization(X[i], y[i], y_pred[i], i)
        else:
            img = process_img_for_visualization(X[i], y[i], None, i)
        displayCV2(img)


def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
    received in RGB)
    '''

    new_img = img[50:140, :, :]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img


def random_distort(img, angle):
    '''
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:, :, 0] + value) > 255
    if value <= 0:
        mask = (new_img[:, :, 0] + value) < 0
    new_img[:, :, 0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h, w = new_img.shape[0:2]
    mid = np.random.randint(0, w)
    factor = np.random.uniform(0.6, 0.8)
    if np.random.rand() > .5:
        new_img[:, 0:mid, 0] *= factor
    else:
        new_img[:, mid:w, 0] *= factor
    # randomly shift horizon
    h, w, _ = new_img.shape
    horizon = 2 * h / 5
    v_shift = np.random.randint(-h / 8, h / 8)
    pts1 = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
    pts2 = np.float32([[0, horizon + v_shift], [w, horizon + v_shift], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    new_img = cv2.warpPerspective(new_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)


def generate_training_data(image_paths, angles, batch_size=128, validation_flag=False):
    '''
    method for the model training data generator to load, process, and distort images, then yield them to the
    model. if 'validation_flag' is true the image is not distorted. also flips images with turning angle magnitudes
    of greater than 0.33, as to give more weight to them and mitigate bias toward low and zero turning angles
    '''
    image_paths, angles = shuffle(image_paths, angles)
    X, y = ([], [])
    while True:
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            img = preprocess_image(img)
            if not validation_flag:
                img, angle = random_distort(img, angle)
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([], [])
                image_paths, angles = shuffle(image_paths, angles)
            # flip horizontally and invert steer angle, if magnitude is > 0.33
            if abs(angle) > 0:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X, y = ([], [])
                    image_paths, angles = shuffle(image_paths, angles)
            # extra flip for adverse steering angles
            if abs(angle) > 0.33:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X, y = ([], [])
                    image_paths, angles = shuffle(image_paths, angles)


def generate_training_data_for_visualization(image_paths, angles, batch_size=20, validation_flag=False):
    '''
    method for loading, processing, and distorting images
    if 'validation_flag' is true the image is not distorted
    '''
    X = []
    y = []
    image_paths, angles = shuffle(image_paths, angles)
    for i in range(batch_size):
        img = cv2.imread(image_paths[i])
        angle = angles[i]
        img = preprocess_image(img)
        if not validation_flag:
            img, angle = random_distort(img, angle)
        X.append(img)
        y.append(angle)
    return (np.array(X), np.array(y))



RunModel = True

My_Data = True
U_Data = True

data_to_use = [My_Data, U_Data]
img_path_prepend = ['', getcwd() + '/data/']
csv_path = ['./data/Recordings/driving_log.csv', './data/driving_log.csv']

image_paths = []
angles = []

for j in range(2):
    if not data_to_use[j]:
        # 0 = my own data, 1 = Udacity supplied data
        print('not using dataset ', j)
        continue
    # Import driving data from csv
    with open(csv_path[j], newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))


    # Gather data - image paths and angles for center, left, right cameras in each row
    for row in driving_data[1:]:
        # skip it if ~0 speed - not representative of driving behavior
        if float(row[6]) < 0.1 :
            continue
        # get center image path and angle
        image_paths.append(img_path_prepend[j] + row[0])
        angles.append(float(row[3]))
        # get left image path and angle
        image_paths.append(img_path_prepend[j] + row[1])
        angles.append(float(row[3])+0.15)
        # get left image path and angle
        image_paths.append(img_path_prepend[j] + row[2])
        angles.append(float(row[3])-0.15)

image_paths = np.array(image_paths)
angles = np.array(angles)

print('Before:', image_paths.shape, angles.shape)

# print a histogram to see which steering angle ranges are most overrepresented
num_bins = 23
avg_samples_per_bin = len(angles)/num_bins
hist, bins = np.histogram(angles, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist, align='center', width=width)
#plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
#plt.show()

# determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
# to number of samples above the average, so as to bring the number of samples for that bin down to the average
keep_probs = []
target = avg_samples_per_bin * .5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(angles)):
    for j in range(num_bins):
        if angles[i] > bins[j] and angles[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
image_paths = np.delete(image_paths, remove_list, axis=0)
angles = np.delete(angles, remove_list)

# print histogram again to show more even distribution of steering angles
hist, bins = np.histogram(angles, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

print('After:', image_paths.shape, angles.shape)

# visualize a single batch of the data
X,y = generate_training_data_for_visualization(image_paths, angles)
visualize_dataset(X,y)

# split into train/test sets
image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles,
                                                                                  test_size=0.05, random_state=42)
print('Train:', image_paths_train.shape, angles_train.shape)
print('Test:', image_paths_test.shape, angles_test.shape)

if RunModel:

    # setup model input and the hard coded normalisation
    inputs = Input(shape=(66,200,3))
    nimg = Lambda(lambda x: x / 127.5 - 1.0)(inputs)

    # -- Start of the NVIDIA portion of the model
    # First 3 convolutional layers with 5x5 kernel each and filter size 24,36 and 48, stride (2,2)
    # ELU activation layers and l2 kernel_regularizers
    x = Conv2D(filters=24,kernel_size=(5,5),strides=(2,2),padding='valid',activation='elu',
               kernel_regularizer=l2(0.001))(nimg)
    x = Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='elu',
               kernel_regularizer=l2(0.001))(x)
    x = Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='elu',
               kernel_regularizer=l2(0.001))(x)
    # End f irst three layers with a dropout layer
    x = Dropout(0.5)(x)

    # Two more convolutional layers with kernel size (3,3) and filter 64
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='elu',
               kernel_regularizer=l2(0.001))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='elu',
               kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    # Add a flatten layer
    x = Flatten()(x)

    # -- Start of the comma.ai model
    # three convolutional layers (note that this branch also starts with the normalised image)
    y = Conv2D(filters=16, kernel_size=(8, 8),strides=(4,4), padding='same', activation='elu')(nimg)
    y = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='elu')(y)
    y = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='elu')(y)

    # Add Dropout layer
    y = Dropout(0.5)(y)

    # Flatten the output
    y = Flatten()(y)

    # Concatenate two models
    z = Concatenate(axis=-1)([x,y])

    # Add three fully connected layers (depth 512, 100, 50, 10), elu activation (and dropouts?)
    z = Dense(512,activation='elu',kernel_regularizer=l2(0.001))(z)
    # add a dropout layer
    z = Dropout(0.5)(z)
    z = Dense(100, activation='elu', kernel_regularizer=l2(0.001))(z)
    z = Dense(50, activation='elu', kernel_regularizer=l2(0.001))(z)
    z = Dense(10, activation='elu', kernel_regularizer=l2(0.001))(z)

    # Add a fully connected output layer
    SteerAngle = Dense(1, activation='elu')(z)

    # Compile and train the model,
    # model.compile
    #print(SteerAngle)
    model = Model(inputs=inputs,outputs=SteerAngle)
    #model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    # initialize generators
    train_gen = generate_training_data(image_paths_train, angles_train, validation_flag=False, batch_size=64)
    val_gen = generate_training_data(image_paths_train, angles_train, validation_flag=True, batch_size=64)
    test_gen = generate_training_data(image_paths_test, angles_test, validation_flag=True, batch_size=64)

    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    history = model.fit_generator(train_gen,epochs=8,steps_per_epoch=2000,verbose=2,callbacks=[checkpoint],
                                  validation_data=val_gen,validation_steps=25)

    print(model.summary())

    # visualize some predictions
    n = 12
    X_test, y_test = generate_training_data_for_visualization(image_paths_test[:n], angles_test[:n], batch_size=n,
                                                              validation_flag=True)
    y_pred = model.predict(X_test, n, verbose=2)
    #visualize_dataset(X_test, y_test, y_pred)

    # Save model data
    model.save_weights('./model_w.h5')
    model.save('model_clone_combo.h5')
    json_string = model.to_json()
    with open('./model.json', 'w') as f:
        f.write(json_string)


