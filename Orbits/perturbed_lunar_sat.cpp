
#include <Tudat/SimulationSetup/tudatSimulationHeader.h>

#include "applicationOutput.h"

//! Execute propagation of orbit of a perturbed satellite around the Moon
//Almost entirely taken from the example programs for a perturbed satellite around the earth but changed to use the moon as central body
//Plus higher harmonic ranks and orers of the gravity field and additional perturbing bodies
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
    using namespace tudat::gravitation;
    using namespace tudat::numerical_integrators;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////     CREATE ENVIRONMENT AND VEHICLE       //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Load Spice kernels.
    spice_interface::loadStandardSpiceKernels( );

    // Set simulation time settings. Did not really touch the start epoch since the time frame i used wont really matter (Jupiter orbits once in this time almost)
    // But useful to check when a general launch epoch is known
    const double simulationStartEpoch = 0.0;
    const double simulationEndEpoch = 4*tudat::physical_constants::JULIAN_YEAR;

    // Define body settings for simulation.
    std::vector< std::string > bodiesToCreate;
    bodiesToCreate.push_back( "Sun" );
    bodiesToCreate.push_back( "Earth" );
    bodiesToCreate.push_back( "Moon" );
    bodiesToCreate.push_back( "Mars" );
    bodiesToCreate.push_back( "Venus" );
    bodiesToCreate.push_back( "Jupiter" );

    // Create body objects.
    std::map< std::string, std::shared_ptr< BodySettings > > bodySettings =
            getDefaultBodySettings( bodiesToCreate, simulationStartEpoch - 300.0, simulationEndEpoch + 300.0 );
    for( unsigned int i = 0; i < bodiesToCreate.size( ); i++ )
    {
        bodySettings[ bodiesToCreate.at( i ) ]->ephemerisSettings->resetFrameOrientation( "J2000" );
        bodySettings[ bodiesToCreate.at( i ) ]->rotationModelSettings->resetOriginalFrame( "J2000" );
    }
    NamedBodyMap bodyMap = createBodies( bodySettings );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE VEHICLE            /////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Create spacecraft object. Can change the mass here, valuable only for the radiation pressure effect
    bodyMap[ "Test" ] = std::make_shared< simulation_setup::Body >( );
    bodyMap[ "Test" ]->setConstantBodyMass( 300.0 );

    // Create radiation pressure settings, used pretty standard coeffecients, impact appears to be minimal
    // Unfortunately the code doesn't like to include multiple eclipsing bodies (both the moon and earth), so it only uses the moon
    // The moon should be the bigger factor anyway
    double referenceAreaRadiation = 1.0;
    double radiationPressureCoefficient = 0.6;
    std::vector< std::string > occultingBodies;
    occultingBodies.push_back( "Moon" );
    std::shared_ptr< RadiationPressureInterfaceSettings > asterixRadiationPressureSettings =
            std::make_shared< CannonBallRadiationPressureInterfaceSettings >(
                "Sun", referenceAreaRadiation, radiationPressureCoefficient, occultingBodies );

    // Create and set radiation pressure settings
    bodyMap[ "Test" ]->setRadiationPressureInterface(
                "Sun", createRadiationPressureInterface(
                    asterixRadiationPressureSettings, "Test", bodyMap ) );

    // Finalize body creation.
    setGlobalFrameBodyEphemerides( bodyMap, "SSB", "J2000" );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////            CREATE ACCELERATIONS          //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Define propagator settings variables.
    SelectedAccelerationMap accelerationMap;
    std::vector< std::string > bodiesToPropagate;
    std::vector< std::string > centralBodies;

    // Define propagation settings. in this case uses the spherical harmonics of the moon up to order and rank 30
    // As well as (as point sources) the bodies of the earth, sun, Mars, Jupiter and venus, and the sun as radiation pressure source
    std::map< std::string, std::vector< std::shared_ptr< AccelerationSettings > > > accelerationsOfAsterix;

    accelerationsOfAsterix[ "Moon" ].push_back( std::make_shared< SphericalHarmonicAccelerationSettings >( 30, 30 ) );

    accelerationsOfAsterix[ "Earth" ].push_back( std::make_shared< AccelerationSettings >(
                                                   basic_astrodynamics::central_gravity ) );
    accelerationsOfAsterix[ "Sun" ].push_back( std::make_shared< AccelerationSettings >(
                                                     basic_astrodynamics::central_gravity ) );
    accelerationsOfAsterix[ "Mars" ].push_back( std::make_shared< AccelerationSettings >(
                                                     basic_astrodynamics::central_gravity ) );
    accelerationsOfAsterix[ "Jupiter" ].push_back( std::make_shared< AccelerationSettings >(basic_astrodynamics::central_gravity ) );
    
    accelerationsOfAsterix[ "Venus" ].push_back( std::make_shared< AccelerationSettings >(
                                                     basic_astrodynamics::central_gravity ) );

    accelerationsOfAsterix[ "Sun" ].push_back( std::make_shared< AccelerationSettings >(
                                                     basic_astrodynamics::cannon_ball_radiation_pressure ) );

    accelerationMap[ "Test" ] = accelerationsOfAsterix;
    bodiesToPropagate.push_back( "Test" );
    centralBodies.push_back( "Moon" );

    basic_astrodynamics::AccelerationMap accelerationModelMap = createAccelerationModelsMap(
                bodyMap, accelerationMap, bodiesToPropagate, centralBodies );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE PROPAGATION SETTINGS            ////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Set Keplerian elements for the orbiter. the 2338 kilometer with 0.038 eccentricity is an example of a (long term) stable orbit
    // i used a high inclination frozen orbit since the lunar module will be injected with a close to polar orbit inclination. therefore a high inclination
    // orbit is significantly more likely to be achievable with constrains on delta v budget.
    Eigen::Vector6d asterixInitialStateInKeplerianElements;
    asterixInitialStateInKeplerianElements( semiMajorAxisIndex ) = 2338.0E3;
    asterixInitialStateInKeplerianElements( eccentricityIndex ) = 0.038;
    asterixInitialStateInKeplerianElements( inclinationIndex ) = unit_conversions::convertDegreesToRadians( 76.5 );
    asterixInitialStateInKeplerianElements( argumentOfPeriapsisIndex ) = unit_conversions::convertDegreesToRadians( 180.2 );
    asterixInitialStateInKeplerianElements( longitudeOfAscendingNodeIndex ) = unit_conversions::convertDegreesToRadians( 269.8 );
    asterixInitialStateInKeplerianElements( trueAnomalyIndex ) = unit_conversions::convertDegreesToRadians( 139.87 );
	
	//set the moon as the central body, didn't change the variable name earthGravitationalParameter, but remapped to lunar paramaters
    double earthGravitationalParameter = bodyMap.at( "Moon" )->getGravityFieldModel( )->getGravitationalParameter( );
    const Eigen::Vector6d asterixInitialState = convertKeplerianToCartesianElements(
                asterixInitialStateInKeplerianElements, earthGravitationalParameter );


    std::shared_ptr< TranslationalStatePropagatorSettings< double > > propagatorSettings =
            std::make_shared< TranslationalStatePropagatorSettings< double > >
            ( centralBodies, accelerationModelMap, bodiesToPropagate, asterixInitialState, simulationEndEpoch );
	
	//set time step size, tested at 1,5 and 10, difference is minimal between these, recommend 10 to reduce computing time
	//standard integrator is RK45
    const double fixedStepSize = 10.0;
    std::shared_ptr< IntegratorSettings< > > integratorSettings =
            std::make_shared< IntegratorSettings< > >
            ( rungeKutta4, 0.0, fixedStepSize );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             PROPAGATE ORBIT            ////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Create simulation object and propagate dynamics.
    SingleArcDynamicsSimulator< > dynamicsSimulator( bodyMap, integratorSettings, propagatorSettings );
    std::map< double, Eigen::VectorXd > integrationResult = dynamicsSimulator.getEquationsOfMotionNumericalSolution( );


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////        PROVIDE OUTPUT TO CONSOLE AND FILES           //////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// set output file name (appears in the same )
    std::string outputSubFolder = "Perturbedmoonsattest/";

    Eigen::VectorXd finalIntegratedState = (--integrationResult.end( ) )->second;
    // Print the position (in km) and the velocity (in km/s) at t = 0.
    std::cout << "Single Moon-Orbiting Satellite." << std::endl <<
                 "The initial position vector of Test is [km]:" << std::endl <<
                 asterixInitialState.segment( 0, 3 ) / 1E3 << std::endl <<
                 "The initial velocity vector of Test is [km/s]:" << std::endl <<
                 asterixInitialState.segment( 3, 3 ) / 1E3 << std::endl;

    // Print the position (in km) and the velocity (in km/s) at t = 4 years.
    std::cout << "After " << simulationEndEpoch <<
                 " seconds, the position vector of Test is [km]:" << std::endl <<
                 finalIntegratedState.segment( 0, 3 ) / 1E3 << std::endl <<
                 "And the velocity vector of Test is [km/s]:" << std::endl <<
                 finalIntegratedState.segment( 3, 3 ) / 1E3 << std::endl;

    // Write perturbed satellite propagation history to file.
    input_output::writeDataMapToTextFile( integrationResult,
                                          "perturbedsattest_a2338_s10_i76.dat",
                                          tudat_applications::getOutputPath( ) + outputSubFolder,
                                          "",
                                          std::numeric_limits< double >::digits10,
                                          std::numeric_limits< double >::digits10,
                                          "," );


    // Final statement.
    // The exit code EXIT_SUCCESS indicates that the program was successfully executed.
    return EXIT_SUCCESS;
}

