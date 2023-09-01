<?php

namespace NumPower\Lattice\Core\Layers;

use NumPower\Lattice\Core\Utils\ArrayValidation;
use NumPower\Lattice\Exceptions\ValueErrorException;

/**
 * Specifies the rank and shape of every input to a layer.
 *
 * Reference:
 *
 *      Keras
 *      https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputSpec
 */
class InputSpec
{
    /**
     * @var ?int
     */
    private ?int $maxNdim;

    /**
     * @var ?int
     */
    private ?int $minNdim;

    /**
     * @var ?string
     */
    private ?string $name;

    /**
     * @var int[]
     */
    private ?array $shape;

    /**
     * @var int|null
     */
    private ?int $ndim;

    /**
     * @var bool|null
     */
    private ?bool $allowLastAxisSqueeze;

    /**
     * @var array|null
     */
    private ?array $axes;

    /**
     * @param array|null $shape
     * @param int|null $ndim
     * @param int|null $maxNdim
     * @param int|null $minNdim
     * @param array|null $axes
     * @param bool $allowLastAxisSqueeze
     * @param string|null $name
     * @throws ValueErrorException
     */
    public function __construct(
        ?array  $shape = null,
        ?int    $ndim = null,
        ?int    $maxNdim = null,
        ?int    $minNdim = null,
        ?array  $axes = null,
        ?bool   $allowLastAxisSqueeze = false,
        ?string $name = null
    )
    {
        $this->setMaxNdim($maxNdim);
        $this->setMinNdim($minNdim);
        $this->setName($name);
        if (isset($shape)) {
            $this->setNdim(count($shape));
            $this->setShape($shape);
        } else {
            $this->setNdim($ndim);
        }
        if (!isset($axes)) {
            $axes = [];
        }
        ArrayValidation::intArrayOrFail("axes", $axes);
        $this->setAxes($axes);
        $this->setAllowLastAxisSqueeze($allowLastAxisSqueeze);
        if ($this->getAxes() && ($this->getNdim() != null || $this->getMaxNdim() != null)) {
            $max_ndim = ($this->getNdim()) ?: $this->getMaxNdim();
            $max_axis = max($this->axes);
            if ($max_axis > $max_ndim) {
                throw new ValueErrorException(
                    "Axis $max_axis is greater than the maximum allowed value: $max_ndim"
                );
            }
        }
    }

    /**
     * @param bool $value
     * @return void
     */
    public function setAllowLastAxisSqueeze(bool $value): void
    {
        $this->allowLastAxisSqueeze = $value;
    }

    /**
     * @param bool $value
     * @return bool
     */
    public function getAllowLastAxisSqueeze(bool $value): bool
    {
        return $this->allowLastAxisSqueeze;
    }

    /**
     * @param ?int[] $shape
     * @return void
     */
    public function setShape(?array $shape): void
    {
        $this->shape = $shape;
    }

    /**
     * @param ?int $ndim
     * @return void
     */
    public function setNdim(?int $ndim): void
    {
        $this->ndim = $ndim;
    }

    /**
     * @param string|null $name
     * @return void
     */
    public function setName(?string $name): void
    {
        $this->name = $name;
    }

    /**
     * @param ?int $minNdim
     * @return void
     */
    public function setMinNdim(?int $minNdim): void
    {
        $this->minNdim = $minNdim;
    }

    /**
     * @param ?int $maxNdim
     * @return void
     */
    public function setMaxNdim(?int $maxNdim): void
    {
        $this->maxNdim = $maxNdim;
    }

    /**
     * @return int|null
     */
    public function getMaxNdim(): ?int
    {
        return $this->maxNdim;
    }

    /**
     * @return int|null
     */
    public function getMinNdim(): ?int
    {
        return $this->minNdim;
    }

    /**
     * @return ?int
     */
    public function getNdim(): ?int
    {
        return $this->ndim;
    }

    /**
     * @return array|null
     */
    public function getShape(): ?array
    {
        return $this->shape;
    }

    /**
     * @return array
     */
    public function getAxes(): array
    {
        return $this->axes;
    }

    /**
     * @param array|null $axes
     * @return void
     */
    private function setAxes(?array $axes): void
    {
        $this->axes = $axes;
    }
}
