# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:34:21 2019

@author: jakisanm
"""

import unittest
import os
working_directory = r'\\C:\Users\jakisanm\Desktop\Analytics\Chicago Crimes'
os.chdir(working_directory)
import ServerStats


class TestServerStats(unittest.TestCase):
    
    def test_add(self):
#        result = SHI.add(10,5)
        self.assertEqual(SHI.add(10,5), 15)
        self.assertEqual(SHI.add(-1,1), 0)
        self.assertEqual(SHI.add(-1,-1), -2)

    def test_substract(self):
#        result = SHI.add(10,5)
        self.assertEqual(SHI.substract(10,5), 5)
        self.assertEqual(SHI.substract(-1,1), -2)
        self.assertEqual(SHI.substract(-1,-1), 0)
        
    def test_multiply(self):
#        result = SHI.add(10,5)
        self.assertEqual(SHI.multiply(10,5), 50)
        self.assertEqual(SHI.multiply(-1,1), -1)
        self.assertEqual(SHI.multiply(-1,-1), 1)
        
    def test_divide(self):
#        result = SHI.add(10,5)
        self.assertEqual(SHI.divide(10,5), 2)
        self.assertEqual(SHI.divide(-1,1), -1)
        self.assertEqual(SHI.divide(-1,-1), 1)
        
        ###Testing to see if the error/Exceptions actually works
        self.assertRaises(ValueError, SHI.divide, 10, 0)
        
        ##Context Manager
        with self.assertRaises(ValueError):
            SHI.divide(10,0)
        
if __name__ == '__main__':
    unittest.main()      
