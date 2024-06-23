/**
 * @file dsd_impl.hpp
 * @author Rukmangadh Sai Myana
 *
 * Implementation of the DSD algorithm for multi-objective optimization
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_DSD_DSD_IMPL_HPP
#define ENSMALLEN_DSD_DSD_IMPL_HPP

#include "dsd.hpp"

namespace ens{

template <typename OptimizerType>
inline DSD<OptimizerType>::DSD(const double lowerBound,
                               const double upperBound) :
    lowerBound(lowerBound),
    upperBound(upperBound)
{
}

template<typename OptimizerType>
DSD<OptimizerType>::DSD(OptimizerType &opt,
                        const double lowerBound,
                        const double upperBound):
    optimizerSingleObjective(opt),
    lowerBound(lowerBound),
    upperBound(upperBound)
{
}

template<typename OptimizerType>
template<typename MatType,
         typename... ArbitraryFunctionType,
         typename... CallbackTypes>
typename MatType::elem_type DSD<OptimizerType>::Optimize(
    std::tuple<ArbitraryFunctionType...>& objectives,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  traits::CheckArbitraryFunctionTypeAPI<ArbitraryFunctionType...,
      BaseMatType>();  // make sure its an arbitrary function
  RequireDenseFloatingPointType<BaseMatType>();  // make sure its dense matrix

  // Specify bounds for all variables separately
  if (lowerBound.n_rows == 1)
    lowerBound = lowerBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);
  if (upperBound.n_rows == 1)
    upperBound = upperBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);

  // Check if we have bounds for all the variables
  assert(lowerBound.n_rows == iterate.n_rows && "The dimensions of "
      "lowerBound are not the same as the dimensions of iterate.");
  assert(upperBound.n_rows == iterate.n_rows && "The dimensions of "
      "upperBound are not the same as the dimensions of iterate.");

  numObjectives = sizeof...(ArbitraryFunctionType);
  numVariables = iterate.n_rows;

  // Useful temporaries
  const BaseMatType castedLowerBound = arma::conv_to<BaseMatType>::from(
      lowerBound);
  const BaseMatType castedUpperBound = arma::conv_to<BaseMatType>::from(
      upperBound);

  std::vector<BaseMatType> anchorCoordinates(
      numObjectives, BaseMatType(iterate.n_rows, iterate.n_cols));
  std::vector<BaseMatType> modifiedAnchorCoordinates(
      numObjectives, BaseMatType(iterate.n_rows. iterate.n_cols));
  
  anchorPoints.set_size(numObjectives, numObjectives);
  modifiedAnchorPoints.set_size(numObjectives, numObjectives);

  Info << "DSD initialized successfully. Optimization started." << std::endl;

  // Compute anchor points for pre-procesor scaling
  ComputeAnchorCoordinates(objectives, anchorCoordinates);
  std::vector<arma::Col<ElemType>> calculatedAnchorPoints(
      numObjectives, arma::Col<ElemType>(numObjectives));
  EvaluateObjectives(anchorCoordinates, objectives, calculatedAnchorPoints);

  // Compute modified anchor points for utopia hyperplane
  ComputeModifiedAnchorCoordinates(objectives, modifiedAnchorCoordinates);
  std::vector<arma::Col<ElemType>> calculatedModifiedAnchorPoints(
      numObjectives, arma::Col<ElemType>(numObjectives));
  EvaluateObjectives(modifiedAnchorCoordinates, objectives,
                     calculatedModifiedAnchorPoints);

  for (int i = 0; i < numObjectives; i++)
  {
    anchorPoints.col(i) =
        arma::conv_to<arma::colvec>::from(calculatedAnchorPoints[i]);
    modifiedAnchorPoints.col(i) =
        arma::conv_to<arma::colvec>::from(calculatedModifiedAnchorPoints[i]);
  }
}

template<typename OptimizerType>
template<std::size_t I, typename MatType, typename... ArbitraryFunctionType>
typename std::enable_if<(I == sizeof...(ArbitraryFunctionType)), void>::type
DSD<OptimizerType>::ComputeAnchorCoordinates(
    std::tuple<ArbitraryFunctionType...>& /* objectives */,
    std::vector<MatType>& /* anchorCoordinates */)
{
  ;// Nothing to do here.
}

template<typename OptimizerType>
template<std::size_t I, typename MatType, typename... ArbitraryFunctionType>
typename std::enable_if<(I < sizeof...(ArbitraryFunctionType)), void>::type
DSD<OptimizerType>::ComputeAnchorCoordinates(
    std::tuple<ArbitraryFunctionType...>& objectives,
    std::vector<MatType>& anchorCoordinates)
{
  optimizerSingleObjective.Optimize(std::get<I>(objectives),
                                    anchorCoordinates.at(I));
  ComputeAnchorPoints<I+1, MatType, ArbitraryFunctionType...>(objectives
      anchorCoordinates);
}

template<typename OptimizerType>
template<std::size_t I, typename MatType, typename... ArbitraryFunctionType>
typename std::enable_if<(I == sizeof...(ArbitraryFunctionType)), void>::type
DSD<OptimizerType>::ComputeModifiedAnchorCoordinates(
    std::tuple<ArbitraryFunctionType...>& /* objectives */,
    std::vector<MatType>& /* modifiedAnchorCoordinates */)
{
  ;// Nothing to do here.
}

template<typename OptimizerType>
template<std::size_t I, typename MatType, typename... ArbitraryFunctionType>
typename std::enable_if<(I < sizeof...(ArbitraryFunctionType)), void>::type
DSD<OptimizerType>::ComputeModifiedAnchorCoordinates(
    std::tuple<ArbitraryFunctionType...>& objectives,
    std::vector<MatType>& modifiedAnchorCoordinates)
{
  optimizerSingleObjective.Optimize(std::get<I>(objectives),
                                    modifiedAnchorCoordinates.at(I));
  ComputeModifiedAnchorPoints<I+1, MatType, ArbitraryFunctionType...>(
      objectives, modifiedAnchorCoordinates);
}

template<typename OptimizerType>
template<std::size_t I, typename MatType, typename... ArbitraryFunctionType>
typename std::enable_if<(I == sizeof...(ArbitraryFunctionType)),
                        typename MatType::elem_type>::type
DSD<OptimizerType>::EvaluateAOF(
    MatType& /* coordinates */,
    std::tuple<ArbitraryFunctionType...>& /* objectives */)
{
  return 0;
}

template<typename OptimizerType>
template<std::size_t I, typename MatType, typename... ArbitraryFunctionType>
typename std::enable_if<(I < sizeof...(ArbitraryFunctionType)), 
                        typename MatType::elem_type>::type
DSD<OptimizerType>::EvaluateAOF(
    MatType& coordinates,
    std::tuple<ArbitraryFunctionType...>& objectives)
{
  return std::get<I>(objectives).Evaluate(coordinates) + 
      EvaluateObjectives<I+1, MatType, ArbitraryFunctionType>(coordinates
          objectives, calculatedObjectives);
}

template<typename OptimizerType>
template<std::size_t I, typename MatType, typename... ArbitraryFunctionType>
typename std::enable_if<(I == sizeof...(ArbitraryFunctionType)), void>::type
DSD<OptimizerType>::EvaluateObjectives(
    std::vector<MatType>& /* coordinates */,
    std::tuple<ArbitraryFunctionType...>& /* objectives */,
    std::vector<arma::Col<typename MatType::elem_type>>&
    /* calculatedObjectives */)
{
  // Nothing to do here.
}

template<typename OptimizerType>
template<std::size_t I, typename MatType, typename... ArbitraryFunctionType>
typename std::enable_if<(I < sizeof...(ArbitraryFunctionType)), void>::type
DSD<OptimizerType>::EvaluateObjectives(
    std::vector<MatType>& coordinates,
    std::tuple<ArbitraryFunctionType...>& objectives,
    std::vector<arma::Col<typename MatType::elem_type>>& calculatedObjectives)
{
  for (int i = 0; i < sizeof...(ArbitraryFunctionType); i++)
  {
    calculatedObjectives[i](I) = std::get<I>(objectives).Evaluate(coordinates[i]);
  }
  EvaluateObjectives<I+1, MatType, ArbitraryFunctionType>(coordinates,
      objectives, calculatedObjectives);
}

}  // namespace ens

#endif /* ENSMALLEN_DSD_DSD_IMPL_HPP */