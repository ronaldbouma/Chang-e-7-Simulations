#include <Tudat/SimulationSetup/tudatSimulationHeader.h>

#include "applicationOutput.h"

//! Execute propagation of orbit of Test around the moon.
int main( )
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////            USING STATEMENTS              //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    using namespace tudat;
    using namespace tudat::simulation_setup;
    using namespace tudat::propagators;
    using namespace tudat::numerical_integrators;
    using namespace tudat::orbital_element_conversions;
    using namespace tudat::basic_mathematics;
    using namespace tudat::unit_conversions;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////     CREATE ENVIRONMENT AND VEHICLE       //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Load Spice kernels.
    spice_interface::loadStandardSpiceKernels( );

    // Set simulation end epoch.
    const double simulationEndEpoch = tudat::physical_constants::JULIAN_DAY;

    // Create body objects.
    std::vector< std::string > bodiesToCreate;
    bodiesToCreate.push_back( "Moon" );
    std::map< std::string, std::shared_ptr< BodySettings > > bodySettings =
            getDefaultBodySettings( bodiesToCreate );
    bodySettings[ "Moon" ]->ephemerisSettings = std::make_shared< ConstantEphemerisSettings >(
                Eigen::Vector6d::Zero( ) );

    // Create Earth object
    NamedBodyMap bodyMap = createBodies( bodySettings );

    // Create spacecraft object.
    bodyMap[ "Testsat" ] = std::make_shared< simulation_setup::Body >( );

    // Finalize body creation.
    setGlobalFrameBodyEphemerides( bodyMap, "SSB", "ECLIPJ2000" );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////            CREATE ACCELERATIONS          //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Define propagator settings variables.
    SelectedAccelerationMap accelerationMap;
    std::vector< std::string > bodiesToPropagate;
    std::vector< std::string > centralBodies;

    bodiesToPropagate.push_back( "Testsat" );
    centralBodies.push_back( "Moon" );

    // Define propagation settings.
    std::map< std::string, std::vector< std::shared_ptr< AccelerationSettings > > > accelerationsOfAsterix;
    accelerationsOfAsterix[ "Moon" ].push_back( std::make_shared< AccelerationSettings >(
                                                     basic_astrodynamics::central_gravity ) );
    accelerationMap[ "Testsat" ] = accelerationsOfAsterix;

    // Create acceleration models and propagation settings.
    basic_astrodynamics::AccelerationMap accelerationModelMap = createAccelerationModelsMap(
                bodyMap, accelerationMap, bodiesToPropagate, centralBodies );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE PROPAGATION SETTINGS            ////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Set initial conditions for the Asterix satellite that will be propagated in this simulation.
    // The initial conditions are given in Keplerian elements and later on converted to Cartesian
    // elements.

    // Set Keplerian elements for Testsat.
    Eigen::Vector6d asterixInitialStateInKeplerianElements;
    asterixInitialStateInKeplerianElements( semiMajorAxisIndex ) = 3000.0E3;
    asterixInitialStateInKeplerianElements( eccentricityIndex ) = 0.25;
    asterixInitialStateInKeplerianElements( inclinationIndex ) = convertDegreesToRadians( 85.3 );
    asterixInitialStateInKeplerianElements( argumentOfPeriapsisIndex ) =
            convertDegreesToRadians( 96.7 );
    asterixInitialStateInKeplerianElements( longitudeOfAscendingNodeIndex ) =
            convertDegreesToRadians( 5.0 );
    asterixInitialStateInKeplerianElements( trueAnomalyIndex ) = convertDegreesToRadians( 139.87 );

    // Convert Asterix state from Keplerian elements to Cartesian elements.
    double earthGravitationalParameter = bodyMap.at( "Moon" )->getGravityFieldModel( )->getGravitationalParameter( );
    Eigen::VectorXd systemInitialState = convertKeplerianToCartesianElements(
                asterixInitialStateInKeplerianElements,
                earthGravitationalParameter );

    // Create propagator settings.
    std::shared_ptr< TranslationalStatePropagatorSettings< double > > propagatorSettings =
            std::make_shared< TranslationalStatePropagatorSettings< double > >
            ( centralBodies, accelerationModelMap, bodiesToPropagate, systemInitialState, simulationEndEpoch );

    // Create numerical integrator settings.
    double simulationStartEpoch = 0.0;
    const double fixedStepSize = 10.0;
    std::shared_ptr< IntegratorSettings< > > integratorSettings =
            std::make_shared< IntegratorSettings< > >
            ( rungeKutta4, simulationStartEpoch, fixedStepSize );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             PROPAGATE ORBIT            ////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Create simulation object and propagate dynamics.
    SingleArcDynamicsSimulator< > dynamicsSimulator( bodyMap, integratorSettings, propagatorSettings );
    std::map< double, Eigen::VectorXd > integrationResult = dynamicsSimulator.getEquationsOfMotionNumericalSolution( );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////        PROVIDE OUTPUT TO CONSOLE AND FILES           //////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::string outputSubFolder = "unperturbedtest/";

    Eigen::VectorXd finalIntegratedState = ( --integrationResult.end( ) )->second;
    // Print the position (in km) and the velocity (in km/s) at t = 0.
    std::cout << "Single Moon-Orbiting Satellite Example." << std::endl <<
                 "The initial position vector of Testsat is [km]:" << std::endl <<
                 systemInitialState.segment( 0, 3 ) / 1E3 << std::endl <<
                 "The initial velocity vector of Testsat is [km/s]:" << std::endl <<
                 systemInitialState.segment( 3, 3 ) / 1E3 << std::endl;

    // Print the position (in km) and the velocity (in km/s) at t = 86400.
    std::cout << "After " << simulationEndEpoch <<
                 " seconds, the position vector of Testsat is [km]:" << std::endl <<
                 finalIntegratedState.segment( 0, 3 ) / 1E3 << std::endl <<
                 "And the velocity vector of Testsat is [km/s]:" << std::endl <<
                 finalIntegratedState.segment( 3, 3 ) / 1E3 << std::endl;

    // Write satellite propagation history to file.
    input_output::writeDataMapToTextFile( integrationResult,
                                          "unperturbedtest.dat",
                                          tudat_applications::getOutputPath( ) + outputSubFolder,
                                          "",
                                          std::numeric_limits< double >::digits10,
                                          std::numeric_limits< double >::digits10,
                                          "," );

    // Final statement.
    // The exit code EXIT_SUCCESS indicates that the program was successfully executed.
    return EXIT_SUCCESS;
}

