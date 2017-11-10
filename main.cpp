/*
A 3D implementation of the SPH SM Monodomain methods.

A cube of tissue simulated by SPH and SM is activated by the monodomain equations.

@author Octavio Navarro
@version 1.0
*/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iostream>
#include <stdio.h>

#include <SPH_SM_monodomain.h>

#if defined(_WIN32) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
// #include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

using namespace std;

struct color{
	float r, g, b;
};

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -5.0;

bool keypressed = false, simulate = true;

float cubeFaces[24][3] = 
{
	{0.0,0.0,0.0}, {1.5,0.0,0.0}, {1.5,1.5,0.0}, {0.0,1.5,0.0},
	{0.0,0.0,0.0}, {1.5,0.0,0.0}, {1.5,0,1.5}, {0.0,0,1.5},
	{0.0,0.0,0.0}, {0,1.5,0.0}, {0,1.5,1.5}, {0.0,0,1.5},
	{0.0,1.5,0.0}, {1.5,1.5,0.0}, {1.5,1.5,1.5}, {0.0,1.5,1.5},
	{1.5,0.0,0.0}, {1.5,1.5,0.0}, {1.5,1.5,1.5}, {1.5,0.0,1.5},
	{0.0,0.0,1.5}, {0,1.5,1.5}, {1.5,1.5,1.5}, {1.5,0,1.5}
};

SPH_SM_monodomain *sph;
int winX = 600;
int winY = 600;

char fps_message[50];
int frameCount = 0;
double average_fps = 0;
float fps = 0;
int total_fps_counts = 0;
int currentTime = 0, previousTime = 0;

const int max_time_steps = 2000;
int time_steps = max_time_steps;
bool simulation_active = true;
duration_d average_step_duration;
tpoint tstart;

void exit_simulation();

void calculateFPS()
{
	//  Increase frame count
	frameCount++;

	//  Get the number of milliseconds since glutInit called
	//  (or first call to glutGet(GLUT ELAPSED TIME)).
	currentTime = glutGet(GLUT_ELAPSED_TIME);

	//  Calculate time passed
	int timeInterval = currentTime - previousTime;

	if (timeInterval > 1000)
	{
		//  calculate the number of frames per second
		fps = frameCount / (timeInterval / 1000.0f);

		if(simulate && simulation_active)
		{
			average_fps += fps;
			total_fps_counts ++;
		}
		//  Set time
		previousTime = currentTime;

		//  Reset frame count
		frameCount = 0;
	}
}

color set_color(float value, float min, float max)
{
	color return_color;
	float ratio = 0.0f;
	float mid_distance = (max - min) / 2;

	if(value <= mid_distance)
	{
		ratio = value / mid_distance;
		return_color.r = ratio;
		return_color.g = ratio;
		return_color.b = (1 - ratio);
	}
	else if(value > mid_distance)
	{
		ratio = (value - mid_distance) / mid_distance;
		return_color.r = 1.0f;
		return_color.g = 1 - ratio;
		return_color.b = 0.0f;
	}
	return return_color;
}

void readCloudFromFile(const char* filename, vector<m3Vector>* points)
{
	FILE *ifp;
	float x, y, z;
	int aux = 0;

	if ((ifp = fopen(filename, "r")) == NULL)
	{
		// fprintf(stderr, "Can't open input file!\n");
		return;
	}

	while ((aux = fscanf(ifp, "%f,%f,%f\n", &x, &y, &z)) != EOF)
	{
		if (aux == 3)
			points->push_back(m3Vector(x,y,z));
	}
}

void display_cube()
{
	glPushMatrix();
		glColor3f(1.0, 1.0, 1.0);
		glLineWidth(2.0);
		for(int i = 0; i< 6; i++)
		{
			glBegin(GL_LINE_LOOP);
			for(int j = 0; j < 4; j++)
				glVertex3f(cubeFaces[i*4+j][0], cubeFaces[i*4+j][1], cubeFaces[i*4+j][2]);
			glEnd();
		}
	glPopMatrix();
}

void display_points()
{
	glPushMatrix();
		Particle *p = sph->Get_Paticles();
		glColor3f(0.2f, 0.5f, 1.0f);
		glPointSize(2.0f);

		glBegin(GL_POINTS);
		for(int i=0; i<sph->Get_Particle_Number(); i++)
		{
			color Voltage_color = set_color(p[i].Vm * sph ->voltage_constant, -200.0f, sph->max_voltage);
			glColor3f(Voltage_color.r, Voltage_color.g, Voltage_color.b);
			glVertex3f(p[i].pos.x, p[i].pos.y, p[i].pos.z);
		}
		glEnd();
	glPopMatrix();
}

void display (void)
{
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	
	glPushMatrix();
		glLineWidth(10.0);
		glBegin(GL_LINES);
		glColor3f(0, 0, 1);
		glVertex3f(0, 0, 0);
		glVertex3f(1, 0, 0);

		glColor3f(1, 0, 0);
		glVertex3f(0, 0, 0);
		glVertex3f(0, 1, 0);

		glColor3f(0, 1, 0);
		glVertex3f(0, 0, 0);
		glVertex3f(0, 0, 1);
		glEnd();
	glPopMatrix();

	display_cube();
	display_points();
	
	glutSwapBuffers();
}

void idle(void)
{
	calculateFPS();
	sprintf(fps_message, "SPH SM M FPS %.3f", fps);
	glutSetWindowTitle(fps_message);

	if(time_steps > 0 && simulate && simulation_active)
	{
		tstart = std::chrono::system_clock::now();
		// cout << "Time Step: " << max_time_steps - time_steps << endl;

		if(time_steps == max_time_steps - max_time_steps / 5)
		{
			sph->turnOffStim();
			// cout << "Turning stimulation off" << endl;
		}

		if(simulate)
		{
			sph->Animation();
			time_steps --;
		}
		average_step_duration += std::chrono::system_clock::now() - tstart;
	}
	else if(time_steps == 0 && simulation_active)
	{
		simulation_active = false;		
		// cout << "Simulation end. Avg step duration: " << average_step_duration.count() / max_time_steps << endl;

		exit_simulation();
	}

	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void exit_simulation()
{
	sph->print_report(average_fps / total_fps_counts, average_step_duration.count() / (max_time_steps - time_steps));
	// cout << "Average FPS: " << average_fps / total_fps_counts<< endl;
	// if(simulation_active)
	// 	cout << "Simulation stopped. Avg step duration: " << average_step_duration.count() / (max_time_steps - time_steps) << endl;
	delete sph;
	exit(0);
}

void keys (unsigned char key, int x, int y)
{
	switch (key) {
		case 27:
			exit_simulation();
			break;
		case 'q':
			sph->turnOffStim();
			// cout << "Turning simulation off" << endl;
			break;
		case 32:
			simulate = !simulate;
			// cout << "Streaming: " << simulate << endl;
			break;
	}
}

void reshape (int w, int h)
{
	glViewport (0,0, w,h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	
	gluPerspective (45, w*1.0/h*1.0, 0.01, 400);
	glMatrixMode (GL_MODELVIEW);
	// glLoadIdentity ();
}

void initGL ()
{
	const GLubyte* renderer;
	const GLubyte* version;
	const GLubyte* glslVersion;

	renderer = glGetString(GL_RENDERER); /* get renderer string */
	version = glGetString(GL_VERSION); /* version as a string */
	glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	// printf("Renderer: %s\n", renderer);
	// printf("OpenGL version supported %s\n", version);
	// printf("GLSL version supported %s\n", glslVersion);

	glEnable(GL_DEPTH_TEST);

	gluLookAt(0.75, 0.75, -translate_z,0.75, 0.75, 0.75, 0, 1, 0);
}

void init_cube()
{
	std::vector<m3Vector> positions;
	m3Vector World_Size = m3Vector(1.0f, 1.0f, 1.0f);
	float kernel = 0.04f;

	for(float k = World_Size.z * 0.3f; k < World_Size.z * 0.7f; k += kernel * 0.6f)
	for(float j = World_Size.y * 0.0f; j < World_Size.y * 0.4f; j += kernel * 0.6f)
	for(float i = World_Size.x * 0.3f; i < World_Size.x * 0.7f; i += kernel * 0.6f)
		positions.push_back(m3Vector(i, j, k));

	sph->Init_Fluid(positions);
	sph->turnOnStim_Cube(positions);
}

void init_mesh(const char * filename)
{
	std::vector<m3Vector> positions;
	readCloudFromFile(filename, &positions);
	sph->Init_Fluid(positions);
	sph->turnOnStim_Mesh(positions);
}

void init(void)
{
	sph = new SPH_SM_monodomain();
	// init_cube();
	init_mesh("Resources/biceps_simple_out.csv");
}

int main( int argc, const char **argv ) {

	srand((unsigned int)time(NULL));
	glutInit(&argc, (char**)argv);
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (winX, winY); 
	glutInitWindowPosition (100, 100);
	glutCreateWindow ("SPH SM 3D");
	glutReshapeFunc (reshape);
	glutKeyboardFunc (keys);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	initGL ();
	init();

	glutDisplayFunc(display); 
	glutIdleFunc (idle);
    glutMainLoop();

	return 0;
}
