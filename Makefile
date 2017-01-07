all: edge face flow hog

edge: edge.cpp
	$(CXX) $< -o $@ `pkg-config opencv --cflags --libs`

face: face.cpp
	$(CXX) $< -o $@ `pkg-config opencv --cflags --libs`

flow: flow.cpp
	$(CXX) $< -o $@ `pkg-config opencv --cflags --libs`

hog: hog.cpp
	$(CXX) $< -o $@ `pkg-config opencv --cflags --libs`

flow-cpu: flow-cpu.cpp
	$(CXX) $< -o $@ `pkg-config opencv --cflags --libs`

clean:
	rm edge face flow hog *~
