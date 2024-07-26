---
layout: post
title: My Journey Learning OpenGL
date: 2024-07-25
subtitle:  draw a point
author: https://yang-alice.github.io/
categories: OpenGL
banner:
  #video: https://vjs.zencdn.net/v/oceans.mp4
  loop: true
  volume: 0.8
  start_at: 8.5
  #image: /assets/images/leetcode/33/Figure_5.png
  opacity: 0.618
  background: "#000"
  height: "100vh"
  min_height: "38vh"
  heading_style: "font-size: 4.25em; font-weight: bold; text-decoration: underline"
  subheading_style: "color: gold"
tags: OpenGL
sidebar: []
---
# get started with OpenGL

Hello, and welcome to my blog! My name is yang, and I am excited to share my journey of learning OpenGL with you. In this series, I will be exploring the world of computer graphics and 3D rendering, starting with the basics and diving into more advanced topics as we progress.

code is pushed to [yang-Alice/openGLPractice github](https://github.com/yang-Alice/openGLPractice/tree/main).

# draw a point

In this tutorial, we will learn how to draw a point using OpenGL. Points are the simplest form of 3D object and are often used as a starting point for more complex shapes. 

Using OpenGL with C++ requires configuring several libraries. You can download the latest versions of the OpenGL, glew and GLFW libraries from their respective websites. For detailed instructions on installing these libraries, refer to "Computer Graphics Programming in OpenGL with C++".


Once you have downloaded the libraries, you can include them in your project using the following code:
```c++
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <iostream>
using namespace std;
```

Next, we create the main function where we will instantiate the GLFW window.
The main() function shown in following Program is the same one that we will use throughout this blog. 

Among the significant operations in main() are:
1. initializes the GLFW library, 
2. instantiates a GLFW window, 
3. initializes the GLEW library, 
4. calls the function “init()” once, 
5. calls the function “display()” repeatedly.

```c++
int main(void) {
	// Initialize the GLFW library
	if (!glfwInit()) { exit(EXIT_FAILURE); }
	// Set the major and minor version of the OpenGL context to 4.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// Create a window with a width and height of 600 pixels, and a title of "point"
	GLFWwindow* window = glfwCreateWindow(600, 600, "point", NULL, NULL);
	// Make the OpenGL context current
	glfwMakeContextCurrent(window);
	// Initialize the GLEW library
	if (glewInit() != GLEW_OK) { exit(EXIT_FAILURE); }
	// Set the swap interval to 1
	glfwSwapInterval(1);

	// Initialize the program
	init(window);

	// Loop until the window is closed
	while (!glfwWindowShouldClose(window)) {
		// Display the scene
		display(window, glfwGetTime());
		// Swap the front and back buffers
		glfwSwapBuffers(window);
		// Poll for and process events
		glfwPollEvents();
	}

	// Destroy the window
	glfwDestroyWindow(window);
	// Terminate the GLFW library
	glfwTerminate();
	// Exit the program
	exit(EXIT_SUCCESS);
}

```


The “init()” function is where we will place application-specific initialization tasks. The display() method is where we place code that draws to the GLFWwindow. we focus on write the init() and display() function in this tutorial.
 


Next, we will create a simple OpenGL program that draws a point. We will use the OpenGL functions to set up the rendering context, create a vertex array object (VAO), and specify the vertex attributes.

To actually draw something, we need to include a vertex shader and a fragment shader.
The vertex shader is responsible for transforming the vertex data and the fragment shader is responsible for coloring the pixels.


In this example, we will use a simple vertex shader that sets the position of the vertex to the origin and a fragment shader that sets the color of the pixel to blue.


The vertex shader is written in GLSL (OpenGL Shading Language) and the fragment shader is written in GLSL as well.


The vertex shader is defined in the createShaderProgram() function and the fragment shader is defined in the createShaderProgram() function.

```c++
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <iostream>
using namespace std;

#define numVAOs 1

GLuint renderingProgram;
GLuint vao[numVAOs];

GLuint createShaderProgram() {
	const char *vshaderSource =
		"#version 430    \n"
		"void main(void) \n"
		"{ gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }";

	const char *fshaderSource =
		"#version 430    \n"
		"out vec4 color; \n"
		"void main(void) \n"
		"{ color = vec4(0.0, 0.0, 1.0, 1.0); }";
	/*
	glCreateShader()函数，创建了类型为GL_VERTEX_ SHADER和GL_FRAGMENT_SHADER的着色器
	*/
	GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);

	/*
	glShaderSource()函数用于将GLSL代码从字符串载入空着色器对象中，并由glCompileShader()编译各着色器。
	glShaderSource()有4个参数：用来存放着色器的着色器对象、着色器源代码中的字符串数量、包含源代码的字符串
	指针，以及一个此处没有用到的参数（我们会在补充说明中解释这个参数）。 
	*/
	glShaderSource(vShader, 1, &vshaderSource, NULL);
	glShaderSource(fShader, 1, &fshaderSource, NULL);
	glCompileShader(vShader);
	glCompileShader(fShader);
	//glCreateProgram()函数创建了一个新的着色器程序对象，并返回该对象的引用
	GLuint vfprogram = glCreateProgram();
	//glAttachShader()函数将着色器对象附加到着色器程序对象上
	glAttachShader(vfprogram, vShader);
	glAttachShader(vfprogram, fShader);
	// 链接着色器程序OpenGL中的着色器程序对象（Shader Program Object）是由多个着色器对象（如顶点着色器、片段着色器等）组合而成的。
	//每个着色器对象在编译后，需要被链接到一个着色器程序对象中，以便形成一个完整的着色器程序，这个程序可以被OpenGL渲染管线使用。
	//glLinkProgram函数就是用来完成这个链接过程的。它接受一个着色器程序对象的句柄作为参数，并尝试将所有已经附加到该程序对象上的着色器对象链接在一起。
	glLinkProgram(vfprogram);

	return vfprogram;
}

void init(GLFWwindow* window) {
	renderingProgram = createShaderProgram();
	glGenVertexArrays(numVAOs, vao);
	glBindVertexArray(vao[0]);
}

void display(GLFWwindow* window, double currentTime) {
	// glUseProgram()，用于将含有两个已编译着色器的程序载入OpenGL管线阶段（在GPU上！）。注意，glUseProgram()并没有运行着色器，它只是将着色器加载进硬件。 
	glUseProgram(renderingProgram);
	glPointSize(30.0f);
	//调用了 glDrawArrays()用来启动管线处理过程。原始类型是GL_POINTS，仅用来显示一个点。 
	glDrawArrays(GL_POINTS, 0, 1);
}

int main(void) {
	if (!glfwInit()) { exit(EXIT_FAILURE); }
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow* window = glfwCreateWindow(600, 600, "point", NULL, NULL);
	glfwMakeContextCurrent(window);
	if (glewInit() != GLEW_OK) { exit(EXIT_FAILURE); }
	glfwSwapInterval(1);

	init(window);

	while (!glfwWindowShouldClose(window)) {
		display(window, glfwGetTime());
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}
```
# link
 
1. [learnopengl official website](https://learnopengl.com/ ): given a lot of examples and tutorials and friendly to beginer.
2. [learnopengl-cn](https://learnopengl-cn.github.io/): the Chinese version of learnopengl, which is more friendly to Chinese people.
3. [opengl-tutorial](https://www.opengl-tutorial.org/): another website that provides a lot of examples and tutorials.
 
 