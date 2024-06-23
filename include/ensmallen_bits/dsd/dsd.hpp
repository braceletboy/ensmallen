/**
 * @file dsd.hpp
 * @author Rukmangadh Sai Myana
 *
 * Directed Search Domain (DSD) is a popular Multi-Objective Optimization(MOO)
 * algorithm. Its a classical MOO optimization method as opposed to
 * evolutionary methods like MOEAD and NSGA. Here, a single objective function
 * called the Aggregate Objective Function (AOF) is optimized in an
 * intelligent manner, leading to computationally cheaper and faster
 * optimization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_DSD_DSD_HPP
#define ENSMALLEN_DSD_DSD_HPP

#include <tuple>
#include <armadillo>

namespace ens{

template<typename OptimizerType>
class DSD
{
 public:
  DSD(const double lowerBound = 0,
      const double upperBound = 1);
  DSD(OptimizerType& opt,
      const double lowerBound = 0,
      const double upperBound = 1);
  DSD(const arma::vec& lowerBound = arma::zeros(1, 1),
      const arma::vec& upperBound = arma::ones(1, 1));
  DSD(OptimizerType& opt,
      const arma::vec& lowerBound = arma::zeros(1, 1),
      const arma::vec& upperBound = arma::ones(1, 1));

  template<typename MatType,
           typename... ArbitraryFunctionType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(
      std::tuple<ArbitraryFunctionType...>& objectives,
      MatType& iterate,
      CallbackTypes&&... callbacks);

  //! Modify the lower bounds
  arma::vec& LowerBound() {return lowerBound;}

  //! Get the lower bounds
  const arma::vec& LowerBound() const {return lowerBound;}

  //! Modify the upper bounds
  arma::vec& UpperBound() {return upperBound;}

  //! Get the upper bounds
  const arma::vec& UpperBound() const {return upperBound;}

  //! Get the anchor points computed - empty until Optimize() called
  const arma::cube& AnchorPoints() const {return anchorPoints;}

  //! Get the modified anchor points computed - empty until Optimize() called
  const arma::cube& ModifiedAnchorPoints() const {return modifiedAnchorPoints;}

 private:

  template<std::size_t I=0, typename MatType,
           typename... ArbitraryFunctionType>
  typename std::enable_if<(I == sizeof...(ArbitraryFunctionType)), void>::type
  ComputeAnchorCoordinates(std::tuple<ArbitraryFunctionType...>& objectives, 
                           std::vector<MatType>& anchorCoordinates);

  template<std::size_t I=0, typename MatType,
           typename... ArbitraryFunctionType,>
  typename std::enable_if<(I < sizeof...(ArbitraryFunctionType)), void>::type
  ComputeAnchorCoordinates(std::tuple<ArbitraryFunctionType...>& objectives,
                           std::vector<MatType>& anchorCoordinates);

  template<std::size_t I=0, typename MatType,
           typename... ArbitraryFunctionType>
  typename std::enable_if<(I == sizeof...(ArbitraryFunctionType)), void>::type
  ComputeModifiedAnchorCoordinates(
      std::tuple<ArbitraryFunctionType...>& objectives,
      std::vector<MatType>& modifiedAnchorCoordinates);

  template<std::size_t I=0, typename MatType,
           typename... ArbitraryFunctionType>
  typename std::enable_if<(I < sizeof...(ArbitraryFunctionType)), void>::type
  ComputeModifiedAnchorCoordinates(
      std::tuple<ArbitraryFunctionType...>& objectives,
      std::vector<MatType>& modifiedAnchorCoordinates);

  template<std::size_t I=0, typename MatType,
           typename... ArbitraryFunctionType>
  typename std::enable_if<(I == sizeof...(ArbitraryFunctionType)),
                          typename MatType::elem_type>::type
  EvaluateAOF(
      MatType& coordinates,
      std::tuple<ArbitraryFunctionType...>& objectives);

  template<std::size_t I=0, typename MatType,
           typename... ArbitraryFunctionType>
  typename std::enable_if<(I < sizeof...(ArbitraryFunctionType)),
                          typename MatType::elem_type>::type
  EvaluateAOF(
      MatType& coordinates,
      std::tuple<ArbitraryFunctionType...>& objectives);

  template<std::size_t I=0, typename MatType,
           typename... ArbitraryFunctionType>
  typename std::enable_if<(I == sizeof...(ArbitraryFunctionType)), void>::type
  EvaluateObjectives(
      std::vector<MatType>& coordinates,
      std::tuple<ArbitraryFunctionType...>& objectives,
      std::vector<arma::Col<typename MatType::elem_type>>&
      calculatedObjectives);

  template<std::size_t I=0, typename MatType,
           typename... ArbitraryFunctionType>
  typename std::enable_if<(I < sizeof...(ArbitraryFunctionType)), void>::type
  EvaluateObjectives(
      std::vector<MatType>& coordinates,
      std::tuple<ArbitraryFunctionType...>& objectives,
      std::vector<arma::Col<typename MatType::elem_type>>&
      calculatedObjectives);

  //! The optimizer to use for single objective optimization steps
  OptimizerType optimizerSingleObjective;

  //! Lower bounds for the variables.
  arma::vec lowerBound;
  
  //! Upper bounds for the variables.
  arma::vec upperBound;

  //! Anchor points for the objectives
  arma::mat anchorPoints;

  //! Modified anchor points for the objectives as described in the algorithm
  arma::mat modifiedAnchorPoints;

};

}  // namespace ens
#endif  /* ENSMALLEN_DSD_DSD_HPP */