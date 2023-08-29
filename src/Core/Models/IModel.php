<?php

namespace NumPower\Lattice\Core\Models;

use NumPower\Lattice\Core\Layers\ILayer;

interface IModel
{
    /**
     * @return ILayer[]
     */
    function getLayers(): array;
}