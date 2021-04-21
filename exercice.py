#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import matplotlib.pyplot as plt


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(start=-1.3, stop=2.5, num=64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:

    # coord_polaire = []
    # for coord in cartesian_coordinates:
    #     x = coord[0]
    #     y = coord[1]
    #     rayon = np.sqrt(x**2 + y**2)
    #     angle = np.arctan2(y, x)
    #     coord_polaire.append((rayon, angle))
    #
    #  np.array(coord_polaire)

    return np.array([(np.sqrt(coord[0]**2 + coord[1]**2), np.arctan2(coord[1], coord[1])) for coord in cartesian_coordinates])


def find_closest_index(values: np.ndarray, number: float) -> int:

    return np.abs(values - number).argmin()

def create_plot():
    x = np.linspace(start=-1, stop=1, num=250)
    y = x**2 * np.sin(1/x**2) + x

    plt.plot(x, y, label="y")
    plt.plot(x, (y**2)/2, label="y**2")
    plt.xlim(-0.5, 0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Exercice chapitre 10")
    plt.legend()
    plt.show()

def monte_carlo(iteration: int=5000) -> float:
    x_inside_circle = []
    x_outside_circle = []

    y_inside_circle = []
    y_outside_circle = []

    for i in range(iteration):
        x = np.random.random()
        y = np.random.random()

        if np.sqrt(x**2 + y**2) <= 1.0:
            x_inside_circle.append(x)
            y_inside_circle.append(y)
        else:
            x_outside_circle.append(x)
            y_outside_circle.append(y)

    s = [4 for n in range(len(x_inside_circle))]
    plt.scatter(x_inside_circle, y_inside_circle, label="Inside Circle", s=s)

    s = [4 for n in range(len(x_outside_circle))]
    plt.scatter(x_outside_circle, y_outside_circle, label="Outside Circle", s=s)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Estimation de la valeur de pi")
    plt.legend()
    plt.show()

    return len(x_inside_circle)/iteration * 4

def monte_carlo2(iteration=5000):
    x = np.random.rand(iteration)
    y = np.random.rand(iteration)
    r = np.sqrt((x ** 2) + (y ** 2))
    ratio = np.count_nonzero(r <= 1)/iteration
    return ratio*4

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    #print(linear_values())

    x = np.array([(2, 4), (4, 6)])
    #print(coordinate_conversion(x))

    #print(find_closest_index(np.array([10,15,20,12,13]), 11.1))

    #create_plot()

    print(monte_carlo(25000))

    #print((monte_carlo2(500000)))