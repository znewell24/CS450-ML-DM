# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:32:02 2019

@author: Zach Newell
"""
import random
import numpy as np

# Part 1
class Movie(object):
    def __init__(self, title = "", year = 0, runtime = 0):
        self.title = title
        self.year = year
        
        if runtime < 0:
            runtime = 0
        self.runtime = runtime
        
    def __repr__(self):
        return "{} ({}) - {} mins".format(self.title, self.year, self.runtime)
    
    def getRuntime(self):
        return (self.runtime // 60), (self.runtime % 60)
    
movie = Movie()
movie.title = "Avatar"
movie.year = 2009
movie.runtime = 162

print(movie)

h, m = movie.getRuntime()
print("hours: {}, minutes: {}".format(h, m))

movie2 = Movie("Star Wars: Episode III- Revenge of the Sith", 2005, 140)
print(movie2)

movie3 = Movie("Negative runtime", 1950, -1)
print(movie3)

# Part 2
def create_movie_list():
    movies = []
    movies.append(Movie("Avatar", 2009, 162))
    movies.append(Movie("Star Wars: Episode III- Revenge of the Sith", 2005, 140))
    movies.append(Movie("Aquaman", 2018, 143))
    movies.append(Movie("Star Wars: Episode VIII- The Last Jedi", 2017, 152))
    
    return movies
    
def main2():
    movies = create_movie_list()
    print("Movies:")
    for movie in movies:
        print(movie)
        
    long_movies = [m for m in movies if m.runtime > 150]

    print("Long Movies:")
    for movie in long_movies:
        print(movie)
    
    ratings = {m.title : random.uniform(0, 5) for m in movies}
    print ("Number of Stars:")
    for title in ratings:
        print("{} - {:.2f} Stars".format(title, ratings[title]))
        
# Part 3
def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100
        
        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

def main3():
    data = get_movie_data()
    
    print(data)
    r = data.shape[0]
    c = data.shape[1]
    print("rows: {}, cols: {}".format(r, c))
    
    print(data[0:2])
    print(data[:,-2,:])
    print(data[:,1])
    
def main():
    main2()
    main3()
    


