Steps:

1) Sampling: The algorithm requires that each gesture and their templates be divided into a set number of points. These gesture points will then be calculated individually to check how close they are to the actual template. For our assignment, we have to create 100 Sample points. I do this by forming a line between the gesture points in order. Then I divide this line into 100 equal pieces.
Reference - https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357)

2) Pruning: In this we are trying to narrow down our sample space. This is done because a simple swipe gesture may encapsulate a lot of letters and hence a lot of words. I have set the Threshold as 15, as anything below will not be enough to register the words, and a large Threshold may take in more words than necessary.

To do this, I have calculated the Start-to-Start distance and End-to-End distance between a template and the input gesture as described in the paper. I have not performed any normalization in this step.

3) Shape Score: We first normalize all the valid template points and valid words. This is done using the get_scaled_points. As described in the paper, it calculates the scaling factor and then multiplies all the points with that factor to scale them down. L here is taken as 1, because I wante dto scale down the shape into a 1x1 cell. L is then divided by the largest side of the rectangle formed by the gesture. 

4) Location Score: Location score is calculated using the given get_delta, get_big_d and get_small_d functions. I then calculated the alpha,by creating a tunnel array as such -> [5,4,3,2,1,1,2,3,4,5] And then dividing them by the total sum so that the sum of the new alpha array is equal to 1.

I didn't do 100 iterations of this as described in the paper. This is because small_d and big_D do not depend on the template. So one iteration is enough.

5) Integration Score: Gave a slight more preference to Location here as compared to shape. I think it performs better.

6) Best Word: As described in the paper, the best integration scores had to be multiplied with the probability of the word. I haven't multiplied by the probability of the word because that is not logical in my opinion. It would have been helpful if it was the probability of the user's past words. A general probability from the language model is bound to create inaccurate results.

Hence, taking N makes no sense. Since we only want to return the best word obtained by the integration score.