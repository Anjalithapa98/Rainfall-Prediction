import React from 'react';
import Home from './pages/Home';

import { Routes, Route, uselocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';

const AnimRoutes = () => {
    const location = uselocation();
    return ( <
        AnimatePresence initial = { true }
        mode = 'wait' >

        <
        Routes key = { location.pathname }
        location = { location } >
        <
        Route path = '/'
        element = { < Home / > }
        /> </Routes >
        <
        /AnimatePresence>

    );
};

export default AnimRoutes;