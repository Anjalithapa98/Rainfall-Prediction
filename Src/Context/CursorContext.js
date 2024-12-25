import React, { useState, useEffect, createContext } from 'react';

export const CursorContext = createContext();

const CursorProvider = ({ children }) => {
    const [cursorPos, setCursorPos] = useState({
        x: 0,
        y: 0,
    });
    const [cursorBG, setCursorBG] = useState('default');

    const mobileViewportIsActive = window.innerWidth < 768;

    useEffect(() => {
        if (mobileViewportIsActive) {


            const move = (e) => {
                setCursorPos({
                    x: e.clientX,
                    y: e.clientY,
                })
            }
            window.addEventListener('mousemove', move);

            return () => {
                window.removeEventListener('mousemove', move);
            };
        } else {
            setCursorBG('none')
        }

    });
    const cursorVariants = {
        default: {
            x: cursorPos.x - 16,
            y: cursorPos.y - 16,
            backgroundcolor: '#0e1112',
        },
        text: {
            width: '150px',
            height: '150px',
            x: cursorPos.x - 72,
            y: cursorPos.y - 72,
            backgroundcolor: '#fff',
            mixBlendMode: 'difference',

        },
        none: {
            width: 0,
            height: 0,
            backgroundcolor: 'rgba(255,255,255,1)',
        },
    };

    const mouseEnterHandler = () => {
        setCursorBG('text');
    }

    const mouseLeaveHandler = () => {
        setCursorBG('deafult');
    }

    return ( < CursorContext.Provider value = {
            { cursorVariants, cursorBG, mouseEnterHandler, mouseLeaveHandler } } > { children } 
    </CursorContext.Provider>
    );
};

export default CursorProvider;
