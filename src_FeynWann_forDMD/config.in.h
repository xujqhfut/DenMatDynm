/*-------------------------------------------------------------------
Copyright 2018 Ravishankar Sundararaman

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#ifndef FEYNWANN_CONFIG_H
#define FEYNWANN_CONFIG_H

//Configuration definitions processed by Cmake
//	Do not edit config.h, it is generated by Cmake
//	Edit config.in.h instead

#include <core/Util.h>

#define PACKAGE_NAME "${CPACK_PACKAGE_NAME}"
#define VERSION_STRING "${VERSION_STRING}"
#define GIT_HASH "${GIT_HASH}"

#endif // FEYNWANN_CONFIG_H
