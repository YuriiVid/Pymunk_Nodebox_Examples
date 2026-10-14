#encoding: utf-8
from __future__ import division
from nodebox.graphics import *
import pymunk
import pymunk.pyglet_util
import random, math
import numpy as np

"""
Лабораторна робота №6: Навчання з підкріпленням (Q-Learning)
Удосконалений модуль керування роботом 8_6.py
Студент: Варіант 4 (a=0, b=2, c=3, d=1)

Основні покращення:
1. Канонічне оновлення Белмана для Q-таблиці (TD Q-learning):
   Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max(Q(s', :)) - Q(s, a))
2. Epsilon-greedy стратегія з експоненційним затуханням для балансу дослідження/експлуатації.
3. Розширений простір станів: 4 стани (чистий простір, ціль s1 у секторі, небезпека s2 у секторі, межа арени).
4. Розширений простір дій: 4 маневри (прямо, доворот на ціль s1, маневр ухилення від s2, розворот до центру).
5. Збалансована функція винагород (Reward Shaping).
"""

space = pymunk.Space()

def createBody(x,y,shape,*shapeArgs):
    body = pymunk.Body()
    body.position = x, y
    s = shape(body, *shapeArgs)
    s.mass = 1
    s.friction = 1
    space.add(body, s)
    return s #shape!!!

s0=createBody(300, 300, pymunk.Poly, ((-20,-5),(-20,5),(20,15),(20,-15)))
s0.score=0

s3=createBody(200, 300, pymunk.Poly, ((-20,-5),(-20,5),(20,15),(20,-15)))
s3.color = (0, 255, 0, 255)
s3.score=0

# Q-таблиця агента: 4 стани x 4 дії
# Стани: 0 - чисто, 1 - ціль s1, 2 - небезпека s2, 3 - небезпека межі арени
# Дії: 0 - прямо, 1 - на ціль s1, 2 - ухилення від s2, 3 - розворот до центру
s3.body.Q = np.zeros((4, 4))
s3.body.action = 0
s3.body.last_state = 0
s3.body.epsilon = 1.0
s3.body.eps_min = 0.05
s3.body.eps_decay = 0.998
s3.body.alpha = 0.15
s3.body.gamma = 0.90

s1=createBody(300, 200, pymunk.Circle, 10, (0,0))
S2=[]
for i in range(1):
    s2=createBody(350, 250, pymunk.Circle, 10, (0,0))
    s2.color = (255, 0, 0, 255)
    S2.append(s2)

def getAngle(x,y,x1,y1):
    return math.atan2(y1-y, x1-x)

def getDist(x,y,x1,y1):
    return ((x-x1)**2+(y-y1)**2)**0.5

def inCircle(x,y,cx,cy,R):
    if (x-cx)**2+(y-cy)**2 < R**2:
        return True
    return False

def inSector(x,y,cx,cy,R,a):
    angle=getAngle(cx,cy,x,y)
    a=a%(2*math.pi)
    angle=angle%(2*math.pi)
    diff = abs((angle - a + math.pi) % (2 * math.pi) - math.pi)
    return inCircle(x, y, cx, cy, R) and (diff < 0.5)

def strategy_baseline(b=s3.body):
    """Базова стратегія з випадковим блуканням для порівняння."""
    v=100
    a=b.angle
    b.velocity=v*cos(a), v*sin(a)
    x,y=b.position
    R=getDist(x,y,350,250)
    if canvas.frame%10==0:
        act = random.choice([0, 1])
        if act == 1:
            b.angle = 2*math.pi*random.random()
        if R > 180:
            b.angle = getAngle(x,y,350,250)

def strategy2(b=s3.body):
    """
    Удосконалена Q-learning стратегія:
    - Визначає стан сенсорів (зір у секторі + наближення до межі);
    - Обирає дію за eps-greedy політикою;
    - Виконує цільове маневрування;
    - Оновлює Q-таблицю за рівнянням Белмана.
    """
    v = 110.0
    a = b.angle
    b.velocity = v * cos(a), v * sin(a)
    x, y = b.position
    R = getDist(x, y, 350, 250)
    
    # Візуалізація сенсорного сектора огляду
    ellipse(x, y, 200, 200, stroke=Color(0.5, 0.5, 0.5, 0.3))
    line(x, y, x + 100 * cos(a + 0.5), y + 100 * sin(a + 0.5), stroke=Color(0.2, 0.8, 0.2, 0.5))
    line(x, y, x + 100 * cos(a - 0.5), y + 100 * sin(a - 0.5), stroke=Color(0.2, 0.8, 0.2, 0.5))

    if canvas.frame % 5 == 0:  # Кожні 5 кадрів
        inS1 = inSector(s1.body.position[0], s1.body.position[1], x, y, 110, a)
        inS2 = inSector(S2[0].body.position[0], S2[0].body.position[1], x, y, 110, a)

        # 1. Визначення стану середовища
        if R > 155:
            current_state = 3  # Небезпека біля стіни
        elif inS1:
            current_state = 1  # Ціль s1 зафіксована
        elif inS2:
            current_state = 2  # Небезпека s2 зафіксована
        else:
            current_state = 0  # Вільний простір

        # 2. Розрахунок винагороди за попередній крок
        reward = 0.0
        if b.last_state == 1 and b.action == 1:
            reward += 1.5  # Заохочення наведення на корисну ціль
        elif b.last_state == 2 and b.action == 2:
            reward += 1.0  # Заохочення ухилення від антиоб'єкта
        elif b.last_state == 3 and b.action == 3:
            reward += 0.8  # Заохочення розвороту від стіни
        else:
            reward += 0.1  # Невелике заохочення за вільний рух

        # 3. Оновлення Q-таблиці за правилом Белмана
        best_next = np.max(b.Q[current_state])
        td_target = reward + b.gamma * best_next
        td_error = td_target - b.Q[b.last_state, b.action]
        b.Q[b.last_state, b.action] += b.alpha * td_error

        # 4. Вибір нової дії (eps-greedy)
        if random.random() < b.epsilon:
            action = random.choice([0, 1, 2, 3])
        else:
            action = int(np.argmax(b.Q[current_state]))
            
        b.epsilon = max(b.eps_min, b.epsilon * b.eps_decay)
        b.action = action
        b.last_state = current_state

        # 5. Виконання обраної цільової дії
        if action == 1:
            # Активний поворот на ціль s1
            b.angle = getAngle(x, y, s1.body.position[0], s1.body.position[1])
        elif action == 2:
            # Маневр ухилення від s2 (відворот по нормалі)
            ang_s2 = getAngle(x, y, S2[0].body.position[0], S2[0].body.position[1])
            b.angle = ang_s2 + math.pi / 2
        elif action == 3 or R > 175:
            # Повернення до центру арени
            b.angle = getAngle(x, y, 350, 250)

def scr(s,s0,s3,p=1):
    bx,by=s.body.position
    s0x,s0y=s0.body.position
    s3x,s3y=s3.body.position
    if not inCircle(bx,by,350,250,180):
        if getDist(bx,by,s0x,s0y)<getDist(bx,by,s3x,s3y):
            s0.score=s0.score+p
        else:
            s3.score=s3.score+p
        s.body.position=random.randint(200,400),random.randint(200,300)

def score():
    """визначає переможця"""
    scr(s1,s0,s3)
    for s in S2:
        scr(s,s0,s3,p=-1)

def manualControl():
    """Керування роботом з мишки або клавіатури"""
    v=10 # швидкість
    b=s0.body
    a=b.angle
    x,y=b.position
    vx,vy=b.velocity
    if canvas.keys.char=="a":
        b.angle-=0.1
    if canvas.keys.char=="d":
        b.angle+=0.1
    if canvas.keys.char=="w":
        b.velocity=vx+v*cos(a), vy+v*sin(a)
    if canvas.mouse.button==LEFT:
        b.angle=getAngle(x,y,*canvas.mouse.xy)
        b.velocity=vx+v*cos(a), vy+v*sin(a)

def simFriction():
    for s in [s0,s1,s3]+S2:
        s.body.velocity=s.body.velocity[0]*0.9, s.body.velocity[1]*0.9
        s.body.angular_velocity=s.body.angular_velocity*0.9

draw_options = pymunk.pyglet_util.DrawOptions()

def draw(canvas):
    canvas.clear()
    fill(0,0,0,1)
    text("Human (s0): %i   |   Q-Robot (s3): %i" % (s0.score, s3.score), 20, 20)
    nofill()
    ellipse(350, 250, 360, 360, stroke=Color(0))
    manualControl()
    strategy2() # Використання вдосконаленої Q-learning стратегії
    score()
    simFriction()
    space.step(0.02)
    space.debug_draw(draw_options)

canvas.size = 700, 500
canvas.run(draw)
