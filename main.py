import math
import matplotlib.pyplot as plt

import pyglet
from pyglet.gl import *
from pyglet.libs.win32.constants import WHITE_BRUSH
from pyglet.window import key, mouse

import pymunk
import pymunk.pyglet_util
from pyglet import shapes
from pymunk import Vec2d

from datetime import datetime
import numpy as np

import yaml

config = None
path = None
err = []

def proj_x_on_y(vec_x, vec_y):
    x = np.array([vec_x.x, vec_x.y])
    y = np.array([vec_y.x, vec_y.y])
    proj = y * np.dot(x, y) / np.dot(y, y)
    return Vec2d(proj[0], proj[1])

def dot_x_y(vec_x, vec_y):
    x = np.array([vec_x.x, vec_x.y])
    y = np.array([vec_y.x, vec_y.y])
    return np.dot(x, y)

class Main(pyglet.window.Window):
    index = 0
    reference_position = None
    batch = pyglet.graphics.Batch()
    
    def __init__(self):
        pyglet.window.Window.__init__(self, vsync=False, height=600, width=600)
        pyglet.gl.glClearColor(1,1,1,0)
        self.set_caption("Simulation of boats")

        pyglet.clock.schedule_interval(self.update, 1 / 60.0)
        self.fps_display = pyglet.window.FPSDisplay(self)

        self.create_world()

        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES


    def create_world(self):
        self.space = pymunk.Space()
        self.space.sleep_time_threshold = 0.3

        # add iceberg
        moment = pymunk.moment_for_circle(config['iceberg']['mass'], 0, config['iceberg']['radius'])
        body = pymunk.Body(config['iceberg']['mass'], moment)
        body.position = Vec2d(config['iceberg']['position']['x'], config['iceberg']['position']['y'])
        shape = pymunk.Circle(body, config['iceberg']['radius'], (0, 0))
        shape.friction = config['world']['friction']
        self.space.add(body, shape)

        # add reference position
        self.reference_position = Vec2d(path['points'][self.index]['x'],path['points'][self.index]['y'])

        # add robots
        moment = pymunk.moment_for_circle(config['robots']['mass'], 0, config['robots']['radius'])
        for position in config['robots']['positions']:
            body = pymunk.Body(config['robots']['mass'], moment)
            body.position = Vec2d(position['x'], position['y'])
            shape = pymunk.Circle(body, config['robots']['radius'], (0, 0))
            shape.friction = config['world']['friction']
            shape.color = (255, 150, 150, 255)
            self.space.add(body, shape)

    def update(self, dt):
        # Attraction (robot->iceberg)
        k_attrac = config['world']['k_a']
        for i in range(len(self.space.bodies)):
            iceberg_position = self.space.bodies[0].position
            if i != 0:
                body_i = self.space.bodies[i]
                direction = (iceberg_position - body_i.position)/abs(iceberg_position - body_i.position)
                velocity_command = k_attrac * direction * (abs(iceberg_position - body_i.position) - (config['robots']['radius'] + config['iceberg']['radius']))
                body_i.velocity += velocity_command
                    

        # Repulsion (body->body)
        min_radius = config['world']['min_radius']
        k_repuls = config['world']['k_r']
        for i in range(len(self.space.bodies)):
            if i != 0:
                body_i = self.space.bodies[i]
                for j in range(len(self.space.bodies)):
                    if j != 0 and j != i:
                        body_j = self.space.bodies[j]
                        r = (body_j.position - body_i.position)
                        if abs(r) < min_radius:
                            velocity_command = - k_repuls * r/abs(r) * (1/abs(r))
                            body_i.velocity += velocity_command
        
        # Projection of iceberg fake attraction
        k_iceberg = config['world']['k_d']
        iceberg = self.space.bodies[0]
        fake_force = k_iceberg * (self.reference_position - iceberg.position)
        for i in range(len(self.space.bodies)):
            if i != 0:
                body_i = self.space.bodies[i]
                r = (iceberg.position - body_i.position)
                if dot_x_y(fake_force, r) > 0:
                    velocity_command = proj_x_on_y(fake_force, r)
                    body_i.velocity += velocity_command

        # Drag force
        for body in self.space.bodies:
            drag = -config['world']['drag']*body.velocity
            angular_drag = -config['world']['angular_drag']*body.angular_velocity
            body.apply_force_at_world_point(drag,body.position)
            body.apply_force_at_world_point(angular_drag*Vec2d(0,1),body.position+Vec2d(1,0))
            body.apply_force_at_world_point(angular_drag*Vec2d(0,-1),body.position-Vec2d(1,0))
        step_dt = 1 / 25.0
        x = 0
        while x < dt:
            x += step_dt
            self.space.step(step_dt)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            mass = 100
            r = 15
            moment = pymunk.moment_for_circle(mass, 0, r, (0, 0))
            body = pymunk.Body(mass, moment)
            body.position = (0, 165)
            shape = pymunk.Circle(body, r, (0, 0))
            shape.friction = 0.3
            shape.color = (255, 150, 150, 255)
            self.space.add(body, shape)
            f = 2000
            body.apply_impulse_at_local_point((f, 0), (0, 0))
        elif symbol == key.ESCAPE:
            pyglet.app.exit()
        elif symbol == key.RIGHT:
            self.reference_position = Vec2d(path['points'][self.index]['x'],path['points'][self.index]['y'])
            self.index = (self.index + 1) % len(path['points'])
            print(f"{self.index}: {self.reference_position}")
        elif symbol == key.LEFT:
            self.reference_position = Vec2d(path['points'][self.index]['x'],path['points'][self.index]['y'])
            self.index = (self.index - 1) % len(path['points'])
            print(f"{self.index}: {self.reference_position}")
        elif symbol == pyglet.window.key.P:
            pyglet.image.get_buffer_manager().get_color_buffer().save(
                f"prints/print_{datetime.now().timestamp()}.png"
            )

    def on_mouse_press(self, x, y, button, modifiers):
        self.space.bodies[1].position = Vec2d(x, y)

    def on_draw(self):
        self.clear()
        self.fps_display.draw()
        self.space.debug_draw(self.draw_options)
        self.draw_path()
        err.append(abs(self.reference_position - self.space.bodies[0].position))
        # self.draw_err()

    def draw_err(self):
        plt.plot(err)
        plt.ylabel('error [m]')
        plt.show()
    
    def draw_path(self):
        pyglet.shapes.Circle(
            self.reference_position[0],
            self.reference_position[1],
            radius = 5,
            color = (255,0,0)).draw()
        for i in range(len(path['points'])-1):
            pyglet.shapes.Line(
                path['points'][i]['x'],
                path['points'][i]['y'],
                path['points'][i+1]['x'],
                path['points'][i+1]['y'],
                1, color=(0,0,0)).draw()

def main():
    global config
    global path
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    with open("path.yml", 'r') as stream:
        path = yaml.safe_load(stream)
    main = Main()
    pyglet.app.run()

if __name__ == '__main__':
    main()