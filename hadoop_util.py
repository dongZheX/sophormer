# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""streaming_reader 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
import os
import sys
import json
import numpy as np
import io
import subprocess


class HadoopUtil(object):
    def __init__(self, hadoop_bin, fs_name, fs_ugi):
        self.hadoop_bin = hadoop_bin
        self.fs_name = fs_name
        self.fs_ugi = fs_ugi

    def ls(self, path):
        """ hdfs_ls """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -ls " + path
        cmd += " | awk '{print $8}'"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        stdout = io.TextIOWrapper(p.stdout, encoding='utf-8')
        filelist = stdout.read().split()
        return filelist
    def ls_len(self, path):
        files = self.ls(path)
        lenlist = []
        for file in files:
            cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
            cmd += " -D hadoop.job.ugi=" + self.fs_ugi
            cmd += " -cat " + file
            cmd += "|wc -l"
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            stdout = io.TextIOWrapper(p.stdout, encoding='utf-8')
            lenlist.append(stdout.read())
        return lenlist
    def open(self, filename):
        """ hdfs_file_open """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -cat " + filename
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    def open_(self, filename):
        """ hdfs_file_open """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -cat " + filename
        return os.popen(cmd)
    def open_head(self, filename, head):
        """ hdfs_file_open """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -cat " + filename +"|head -n "+str(head)
        return os.popen(cmd)
    def open_tail(self, filename, tail):
        """ hdfs_file_open """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -cat " + filename +"|tail -n "+str(tail)
        return os.popen(cmd)
    def open_sed(self, filename, start, end):
        """ hdfs_file_open """
        cmd = self.hadoop_bin + " fs -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -cat " + filename +"|sed -n "+str(start)+","+str(end)+"p"
        return os.popen(cmd)
    def get(self, dst_file, src_file):
        cmd = self.hadoop_bin + " fs -Ddfs.client.block.write.retries=15 -Ddfs.rpc.timeout=300000 -Ddfs.delete.trash=1 -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -get " + dst_file + " " + src_file + " 2>>hdfs.err "
        ret = os.system(cmd)
        return ret
    
    def upload(self, src_file, dst_file):
        cmd = self.hadoop_bin + " fs -Ddfs.client.block.write.retries=15 -Ddfs.rpc.timeout=300000 -Ddfs.delete.trash=1 -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -put " + src_file + " " + dst_file + " 2>>hdfs.err "
        ret = os.system(cmd)
        return ret
    
    def rmr(self, dst_file):
        cmd = self.hadoop_bin + " fs -Ddfs.client.block.write.retries=15 -Ddfs.rpc.timeout=300000 -Ddfs.delete.trash=1 -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -rmr " + dst_file
        ret = os.system(cmd)
        return ret
    def mv(self, src, dst):
        cmd = self.hadoop_bin + " fs -Ddfs.client.block.write.retries=15 -Ddfs.rpc.timeout=300000     -Ddfs.delete.trash=1 -D fs.default.name=" + self.fs_name
        cmd += " -D hadoop.job.ugi=" + self.fs_ugi
        cmd += " -mv " + src +" " + dst
        ret = os.system(cmd)
