#!/bin/bash
for i in $(find tamkin test | egrep "\.pyc$|\.py~$|\.pyc~$|\.bak$") ; do rm -v ${i}; done

rm -vr debian/python-*
rm -vr debian/pycompat
rm -vr debian/compat
rm -vr debian/files
rm -vr debian/stamp-makefile-build
rm -vr python-build-stamp-* 

rm -vr test/tmp
rm -vr test/output/*

rm -v MANIFEST
rm -vr dist
rm -vr build
rm -vr doc/_build
rm -vr doctrees
