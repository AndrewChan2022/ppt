//
// Copyright 2023-2024 Chen Kai
//
// This program is free software: you can redistribute it and/or modify it under the terms 
// of the GNU General Public License as published by the Free Software Foundation, either 
// version 3 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with this program. 
// If not, see <https://www.gnu.org/licenses/>.
//

//
// Implement 
//

#pragma once
#include <memory>


namespace feynman {

class Noncopyable : public std::enable_shared_from_this<Noncopyable> {

public:
    Noncopyable();
    virtual ~Noncopyable();

    // no copy
    Noncopyable(Noncopyable const &) = delete;
    Noncopyable& operator=(Noncopyable const &) = delete;

    // no move
    Noncopyable(Noncopyable&&) = delete;
    Noncopyable& operator=(Noncopyable&&) = delete;

    static void printEndFrameStat();
    static void printEndAppStat();
};

} // namespace feynman
