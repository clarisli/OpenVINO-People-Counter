from collections import deque
import time
import math
import logging as log

from yolo import YoloDetector

#log.basicConfig(level=log.DEBUG)

class PeopleCounter:

    class Person:
    
        def __init__(self, obj, start_time=time.time()):
            self.centroidx, self.centroidy = self._get_centroid(obj)
            self.start_time = start_time
            self.last_occurance_time = start_time
            
        def get_duration(self):
            return self.last_occurance_time - self.start_time
        
        def _get_centroid(self, obj):
            centroidx, centroidy = (obj['xmin'] + obj['xmax'])*1.0/2, (obj['ymin'] + obj['ymax'])*1.0/2
            return centroidx, centroidy

        def update(self, obj):
            self.centroidx, self.centroidy = self._get_centroid(obj)
            self.last_occurance_time = time.time()

        def get_distance(self, obj):
            centroidx, centroidy = self._get_centroid(obj)
            distance = math.sqrt(math.pow(centroidx - self.centroidx, 2) +  math.pow(centroidy - self.centroidy, 2) * 1.0) 
            return distance

        


    def __init__(self, dist_threshold):
        self.total_count = 0
        self.current_count = 0
        self.people = []
        self.dist_threshold = dist_threshold

    def is_new_entry(self, objects):
        new_count = self._get_new_count(objects)
        self.total_count += new_count
        return new_count > 0

    def _get_new_count(self, objects):
        count = 0
        for obj in objects:
            if self._is_new_person(obj):
                count += 1
                new_person = PeopleCounter.Person(obj, time.time())
                self.people.append(new_person)
        return count

    
    def _is_new_person(self, obj):
        for i in range(len(self.people)):
            person = self.people[i]
            distance = person.get_distance(obj)
            if distance < self.dist_threshold:
                self.people[i].update(obj)
                return False
        return True

    def get_new_exit_durations(self):
        now = time.time()
        durations = []
        for person in self.people:
            if now - person.last_occurance_time > 0.5:
                durations.append(person.get_duration())
                self.people.remove(person)
        return durations