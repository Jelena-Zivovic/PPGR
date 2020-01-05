from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import math
import functions as f 

positionCenter1 = [8, -1, -4]  #begin
positionCenter2 = [-4, 0, 7]  #end

eulerAngles1 = [math.pi/2, math.pi/3, math.pi/4]
eulerAngles2 = [math.pi/2, math.pi/4, 2*math.pi/3]

q = []
q1 = []
q2 = []

t = 0
tm = 30

TIMER_ID = 0
TIMER_INTERVAL = 20

animation_ongoing = False

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(700, 600)
    glutCreateWindow("slerp")
    
    glLineWidth(2)
    
    glutKeyboardFunc(keyboard)
    glutDisplayFunc(display)
    if animation_ongoing:
        glutTimerFunc(TIMER_INTERVAL, timer, TIMER_ID)
    
    
    glClearColor(0.25, 0.25, 0.25, 1)
    glEnable(GL_DEPTH_TEST)
    
    global q1
    global q2
    
    A = f.Euler2A(eulerAngles1[0], eulerAngles1[1], eulerAngles1[2])
    p, angle = f.AxisAngle(A)
    q1 = f.AngleAxis2Q(p, angle)
    
    A = f.Euler2A(eulerAngles2[0], eulerAngles2[1], eulerAngles2[2])
    p, angle = f.AxisAngle(A)
    q2 = f.AngleAxis2Q(p, angle)
    
    
    glutMainLoop()
    return


def keyboard(key, x, y):
    
    global animation_ongoing
    
    if ord(key) == 27:
        sys.exit(0)
        
    if ord(key) == ord('g'):
        if not animation_ongoing:
            glutTimerFunc(TIMER_INTERVAL, timer, TIMER_ID)
            animation_ongoing = True
        animation_ongoing = True
    
    if ord(key) == ord('s'):
        animation_ongoing = False
            
            
def timer(value):
    if value != TIMER_ID:
        return
    
    global t
    global tm 
    global animation_ongoing
    global q
    
    t += 0.2
        
    if t >= tm:
        t = 0
        animation_ongoing = False
        return
    
    glutPostRedisplay()
    
    if animation_ongoing:
        glutTimerFunc(TIMER_INTERVAL, timer, TIMER_ID)

def draw_cube(position, angles):
    glPushMatrix()
    
    glColor3f(1, 1, 0)
    
    glTranslatef(position[0], position[1], position[2])
    
    A = f.Euler2A(angles[0], angles[1], angles[2])
    p, angle = f.AxisAngle(A)
    
    glRotatef(angle/math.pi*180, p[0], p[1], p[2])
    glutWireCube(1)
    
    draw_axis(2)
    
    glPopMatrix()

def animate():
    global q
    global t
    global tm
    
    glPushMatrix()
    glColor3f(1, 1, 0)
    
    position = []
    
    position.append((1-t/tm)*positionCenter1[0] + (t/tm)*positionCenter2[0])
    position.append((1-t/tm)*positionCenter1[1] + (t/tm)*positionCenter2[1])
    position.append((1-t/tm)*positionCenter1[2] + (t/tm)*positionCenter2[2])

    glTranslatef(position[0], position[1], position[2])
    
    q = f.slerp(q1, q2, tm, t)
    
    p, angle = f.Q2AxisAngle(q)
    
    glRotatef(angle/math.pi*180, p[0], p[1], p[2])
    
    glutWireCube(1)
    draw_axis(2)
    
    glPopMatrix()
    
def draw_axis(size):
    
    glBegin(GL_LINES)
    glColor3f(1, 0, 0)
    glVertex3f(0,0,0)
    glVertex3f(size,0,0)
        
    glColor3f(0,1,0)
    glVertex3f(0,0,0)
    glVertex3f(0,size,0)
        
    glColor3f(0,0,1)
    glVertex3f(0,0,0)
    glVertex3f(0,0,size)
    
    glEnd()
    

def display():
    
    global q
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    window_width = 700
    window_height = 600
    
    glViewport(0, 0, window_width, window_height)
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60,float( window_width) /  window_height, 1, 25)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(10, 10, 10, 0, 0, 0, 0, 1, 0)
    
    draw_axis(15) 
    
    draw_cube(positionCenter1, eulerAngles1)
    draw_cube(positionCenter2, eulerAngles2) 
    
    animate()
    
    glutSwapBuffers()


if __name__ == '__main__': 
    main()