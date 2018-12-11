#(@) $Id: pyrdb.py,v 3.0 2007/01/17 13:22:12 vltsccm Exp $
#
# who       when      what
# --------  --------  ----------------------------------------------
# dqueloz       27/10/05


import os,sys,string,pickle
from numpy import *


def read_rdb(filename):

    f = open(filename, 'r')
    data = f.readlines()
    f.close()

    key = string.split(data[0][:-1],'\t')
    output = {}
    for i in range(len(key)): output[key[i]] = []

    for line in data[2:]:
        qq = string.split(line[:-1],'\t')
        for i in range(len(key)):
            try: value = float(qq[i])
            except ValueError: value = qq[i]
            output[key[i]].append(value)

    return output


def write_rdb(filename,data,keys,format):

    f = open(filename, 'w')

    head1 = string.join(keys,'\t')
    head2 = ''
    for i in head1:
        if i=='\t': head2 = head2+'\t'
        else: head2 = head2+'-'

    f.write(head1+'\n')
    f.write(head2+'\n')

    if len(data.values()) > 0:
        for i in range(len(data.values()[0])):
            line = []
            for j in keys: line.append(data[j][i])
            f.write(format % tuple(line))

    f.close()


def read_rdb_rows(filename,refcol):

    f = open(filename, 'r')
    data = f.readlines()
    f.close()

    key = string.split(data[0][:-1],'\t')
    iref = key.index(refcol)
    output = {}

    for line in data[2:]:
        qq1 = string.split(line[:-1],'\t')
        qq2 = {}
        for i in range(len(key)): qq2[key[i]] = qq1[i]
        output[qq1[iref]] = qq2

    return output

def ajustement_lenght_for_write(vecteur,max_len):
    vecteur_write=array([1e30]*max_len)
    for i in arange(0,len(vecteur),1):
        vecteur_write[i]=vecteur[i]
    return vecteur_write


def read_to_long_vecteur(vecteur):
    for i in arange(0,len(vecteur),1):
        if vecteur[i]==1e30:
            i=i-1
            break
    vecteur=vecteur[:i+1]
    return vecteur
