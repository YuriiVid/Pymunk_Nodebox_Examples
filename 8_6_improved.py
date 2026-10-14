# -*- coding: utf-8 -*-
"""
Удосконалена стратегія робота на основі навчання з підкріпленням (Q-Learning)
для середовища Pymunk / Nodebox (модуль 8_6_improved.py).

Основні вдосконалення порівняно з базовою версією 8_6.py:
1. Застосовано класичне рівняння Белмана для оновлення Q-таблиці з дисконтуванням майбутніх винагород:
   Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', :)) - Q(s, a))
2. Впроваджено eps-greedy стратегію з поступовим затуханням (epsilon-decay) для оптимального
   балансу між дослідженням середовища (Exploration) та експлуатацією досвіду (Exploitation).
3. Розширено простір дій цілеспрямованими маневрами:
   - Action 0: Збереження поточної швидкості та курсу (рух вперед);
   - Action 1: Активний доворот на корисну ціль (s1) для її захоплення та виштовхування;
   - Action 2: Маневр ухилення від антиоб'єкта (s2);
   - Action 3: Повернення до центру поля при небезпечному наближенні до межі арени (R > 160).
4. Простір станів розширено врахуванням як секторного сканування, так і близькості до периметра арени.
"""

from __future__ import division
import math
import random
import numpy as np

# Спроба імпорту графічних бібліотек (для автономного запуску з GUI)
try:
    from nodebox.graphics import *
    import pymunk
    import pymunk.pyglet_util
    HAS_GUI = True
except ImportError:
    HAS_GUI = False

class AdvancedQRobotPolicy:
    def __init__(self, alpha=0.15, gamma=0.92, eps_start=1.0, eps_min=0.05, eps_decay=0.995):
        self.alpha = alpha      # Швидкість навчання
        self.gamma = gamma      # Коефіцієнт дисконтування
        self.epsilon = eps_start # Ймовірність випадкової дії
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        
        # Стани:
        # 0: Вільний сектор (об'єктів немає)
        # 1: Ціль s1 виявлена у секторі
        # 2: Антиоб'єкт s2 виявлений у секторі
        # 3: Небезпека вильоту за межі арени (R > 160)
        #
        # Дії:
        # 0: Рух прямо
        # 1: Доворот на корисну ціль (s1)
        # 2: Ухилення від антиоб'єкта (s2)
        # 3: Повернення до центру арени
        self.Q = np.zeros((4, 4))
        self.last_state = 0
        self.last_action = 0
        
    def choose_action(self, state):
        """Вибір дії за eps-greedy політикою"""
        if random.random() < self.epsilon:
            action = random.choice([0, 1, 2, 3])
        else:
            action = int(np.argmax(self.Q[state]))
            
        # Затухання epsilon
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        return action
        
    def update(self, state, action, reward, next_state):
        """Оновлення Q-таблиці за правилом Белмана"""
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

# Глобальний екземпляр нової політики
robot_policy = AdvancedQRobotPolicy()

def getAngle(x, y, x1, y1):
    return math.atan2(y1 - y, x1 - x)

def getDist(x, y, x1, y1):
    return math.hypot(x - x1, y - y1)

def inCircle(x, y, cx, cy, R):
    return (x - cx)**2 + (y - cy)**2 < R**2

def inSector(x, y, cx, cy, R, a):
    angle = getAngle(cx, cy, x, y)
    a = a % (2 * math.pi)
    angle = angle % (2 * math.pi)
    diff = abs((angle - a + math.pi) % (2 * math.pi) - math.pi)
    return inCircle(x, y, cx, cy, R) and (diff < 0.5)

def advanced_strategy(b, s1_body, s2_body, center=(350, 250), arena_R=180):
    """
    Удосконалена функція керування роботом b у середовищі PyMunk.
    Інтегрує аналіз сенсорів, вибір дій навченим агентом та оновлення знань.
    """
    v = 110.0
    x, y = b.position
    a = b.angle
    dist_to_center = getDist(x, y, center[0], center[1])
    
    # 1. Сенсорний аналіз середовища
    in_s1 = inSector(s1_body.position[0], s1_body.position[1], x, y, 110, a)
    in_s2 = inSector(s2_body.position[0], s2_body.position[1], x, y, 110, a)
    
    # 2. Визначення поточного стану агента
    if dist_to_center > arena_R - 25:
        current_state = 3  # Небезпека біля межі
    elif in_s1:
        current_state = 1  # Ціль у секторі
    elif in_s2:
        current_state = 2  # Небезпека у секторі
    else:
        current_state = 0  # Чистий рух
        
    # 3. Вибір дії інтелектуальним агентом
    action = robot_policy.choose_action(current_state)
    
    # 4. Виконання вибраної дії
    reward = 0.0
    if action == 0:
        # Рух прямо
        reward += 0.1
    elif action == 1:
        # Активний поворот на ціль s1
        b.angle = getAngle(x, y, s1_body.position[0], s1_body.position[1])
        reward += 1.5 if in_s1 else -0.5
    elif action == 2:
        # Маневр ухилення від s2 (поворот на 90 градусів від напрямку на антиоб'єкт)
        angle_s2 = getAngle(x, y, s2_body.position[0], s2_body.position[1])
        b.angle = angle_s2 + math.pi / 2
        reward += 1.0 if in_s2 else -0.2
    elif action == 3:
        # Повернення до безпечного центру
        b.angle = getAngle(x, y, center[0], center[1])
        reward += 0.8 if dist_to_center > arena_R - 25 else -0.5
        
    # Застосування вектора швидкості
    b.velocity = v * math.cos(b.angle), v * math.sin(b.angle)
    
    # 5. Оновлення Q-таблиці
    robot_policy.update(robot_policy.last_state, robot_policy.last_action, reward, current_state)
    robot_policy.last_state = current_state
    robot_policy.last_action = action

if __name__ == '__main__':
    print("Модуль 8_6_improved.py успішно ініціалізовано.")
    print("Поточна матриця Q-значень агента (форма 4x4):\\n", robot_policy.Q)
    print("Рівень дослідження epsilon:", robot_policy.epsilon)
