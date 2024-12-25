/* eslint-disable no-unreachable */
import React, { useContext } from 'react';
import rainfalllogo from '..rainfalllogo.jpg';
import { Link } from 'react-router-dom';
import MobileNav from './MobileNav';
import { CursorContext } from '../context/CursorContext';

const Header = () => {
    const { mouseEnterHandler, mouseLeaveHandler } = useContext(CursorContext)

    return <header className = 'bg-pink-200 ficed w-full px-[30px] lg:px-[100px] z-30 h-[100px] 
    lg: h-[140 px] flex items-enter ' > Header < /header>;
        <div className = 'flex flex-col lg:flex-row lg:items-center w-full justify-between' >

        <Link
onMouseEnter = { mouseEnterHandler }
    onMouseLeave = { mouseLeaveHandler }
    to = { '/' }
    className = 'max-w-[200px]' >
        <img src = { rainfalllogo }
    alt = '' />

        </Link> <
    nav className = 'hidden xl:flex gap-x-12 font-semibold' > nav
        <Link
    onMouseEnter = { mouseEnterHandler }
    onMouseLeave = { mouseLeaveHandler }
    to = { '/' }
    className = 'text-[#696c6d] hover:text-primary transition' > Home < /Link>

    </nav> < MobileNav />
        </div>
};

export default Header;
