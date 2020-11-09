from flask import Flask, request
from flask import render_template
import time
import json
from scipy.interpolate import interp1d
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''


    '''Calculating the length of the line formed by the input coorinates and then dividing them into 100 equal parts (ref - https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357)'''

    # Linear length on the line
    distance = np.cumsum(np.sqrt( np.ediff1d(points_X, to_begin=0)**2 + np.ediff1d(points_Y, to_begin=0)**2 ))
    distance = distance/distance[-1]

    fx, fy = interp1d( distance, points_X ), interp1d( distance, points_Y  )

    #Dividing the line into 100 sample points, each at equal distance from each other
    alpha = np.linspace(0, 1, 100)
    sample_points_X, sample_points_Y  = fx(alpha), fy(alpha)

    return sample_points_X, sample_points_Y


template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider reasonable)
    to narrow down the number of valid words so that ambiguity can be avoided.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_probabilities, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], [], []

    threshold = 15

    for i, (sample_word_x,sample_word_y) in enumerate(zip(template_sample_points_X,template_sample_points_Y)):

        #Template Sample Points contain [[100 points] for each word in the dic]
        #Sample word contains the gesture for that word divided into 100 points

        sample_start_x = sample_word_x[0]
        sample_start_y = sample_word_y[0]
        sample_end_x = sample_word_x[-1]
        sample_end_y = sample_word_y[-1]

        gesture_start_x = gesture_points_X[0][0]
        gesture_start_y = gesture_points_Y[0][0]
        gesture_end_x = gesture_points_X[0][-1]
        gesture_end_y = gesture_points_Y[0][-1]

        start_to_start = ((sample_start_x - gesture_start_x)**2 + (sample_start_y - gesture_start_y)**2)**0.5

        end_to_end = ((sample_end_x - gesture_end_x)**2 + (sample_end_y -gesture_end_y)**2)**0.5

        if start_to_start <= threshold and end_to_end <= threshold:
            valid_words.append(words[i])
            valid_probabilities.append(probabilities[words[i]])
            valid_template_sample_points_X.append(sample_word_x)
            valid_template_sample_points_Y.append(sample_word_y)

    return valid_words, valid_probabilities, valid_template_sample_points_X,valid_template_sample_points_Y


def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_maximum = max(sample_points_X)
    x_minimum = min(sample_points_X)
    W = x_maximum - x_minimum
    y_maximum = max(sample_points_Y)
    y_minimum = min(sample_points_Y)
    H = y_maximum - y_minimum
    r = L/max(H, W)

    gesture_X, gesture_Y = [], []
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(r * point_x)
        gesture_Y.append(r * point_y)

    centroid_x = (max(gesture_X) - min(gesture_X))/2
    centroid_y = (max(gesture_Y) - min(gesture_Y))/2
    scaled_X, scaled_Y = [], []
    for point_x, point_y in zip(gesture_X, gesture_Y):
        scaled_X.append(point_x - centroid_x)
        scaled_Y.append(point_y - centroid_y)

    return scaled_X, scaled_Y

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''

    L = 1

    shape_scores = []
    if len(valid_template_sample_points_X) == 0 or len(valid_template_sample_points_Y) == 0:
        return shape_scores

    #Scale the gesture points
    scaled_gesture_points_X, scaled_gesture_points_Y = get_scaled_points( gesture_sample_points_X[0], gesture_sample_points_Y[0], L)

    #Scale the each template. Each template has 100 points
    scaled_template_points_X = []
    scaled_template_points_Y = []

    for template_points_X, template_points_Y in zip(valid_template_sample_points_X, valid_template_sample_points_Y):
        scaled_X, scaled_Y = get_scaled_points(template_points_X,template_points_Y,L)
        scaled_template_points_X.append(scaled_X)
        scaled_template_points_Y.append(scaled_Y)

    #Use Propotional Matcher like in Paper
    for template_points_X, template_points_Y in zip(scaled_template_points_X,scaled_template_points_Y):
        sum = 0
        for i in range(100):
            distance = ( (scaled_gesture_points_X[i] - template_points_X[i])**2 + (scaled_gesture_points_Y[i] - template_points_Y[i])**2 )**0.5

            sum += distance
        
        sum = sum/100

        shape_scores.append(sum)

    return shape_scores

def get_small_d(p_X, p_Y, q_X, q_Y):
    min_distance = []
    for n in range(0, 100):
        distance = math.sqrt((p_X - q_X[n])**2 + (p_Y - q_Y[n])**2)
        min_distance.append(distance)
    return (sorted(min_distance)[0])

def get_big_d(p_X, p_Y, q_X, q_Y, r):
    final_max = 0
    for n in range(0, 100):
        local_max = 0
        distance = get_small_d(p_X[n], p_Y[n], q_X, q_Y)
        local_max = max(distance-r , 0)
        final_max += local_max
    return final_max

def get_delta(u_X, u_Y, t_X, t_Y, r, i):
    D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
    D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
    if D1 == 0 and D2 == 0:
        return 0
    else:
        return math.sqrt((u_X[i] - t_X[i])**2 + (u_Y[i] - t_Y[i])**2)

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''

    location_scores = []
    radius = 15

    if len(valid_template_sample_points_X) == 0 or len(valid_template_sample_points_Y) == 0:
        return location_scores

    #Implementing Alpha as decribed in the paper. Decreasing from the ends to the middle like a tunnel.

    alpha = []
    
    for i in range(1,51):
        alpha.append(i)

    alpha = alpha[::-1] + alpha

    SUM = sum(alpha)
    for i in range(len(alpha)):
        alpha[i] /= SUM

    for template_points_X, template_points_Y in zip(valid_template_sample_points_X, valid_template_sample_points_Y):

        #Not doing 100 iterations as in the paper. This is because small_d and big_D do not depend on the template. So one iteration is enough.

        delta = get_delta(gesture_sample_points_X[0], gesture_sample_points_Y[0], template_points_X, template_points_Y, radius, i)
        prod = delta * alpha[i]
        location_scores.append(prod)

    return location_scores

def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.3
    # TODO: Set your own location weight
    location_coef = 0.7
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = 5
    # TODO: Get the best word 

    #Not making use of the Probabilities, as I feel they tend to give rather inaccurate results.

    for word, score in sorted(zip(valid_words,integration_scores), key = lambda x:x[1]):
        best_word = word
        best_score = score
        break
    
    return best_word     


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_probabilities, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()


    print('{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}')

    return '{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}'


if __name__ == "__main__":
    app.run()
